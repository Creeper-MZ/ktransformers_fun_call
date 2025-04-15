# Import necessary libraries
# Corrected: Added Tuple to imports
from typing import Any, AsyncIterable, List, Optional, Set, Dict, Tuple
from ktransformers.models.custom_cache import KDeepSeekV3Cache # Assuming this exists and is relevant
from transformers import (
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
    StaticCache, # Keep StaticCache import if used elsewhere, but KDeepSeekV3Cache seems primary
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from ktransformers.server.config.config import Config
from ktransformers.server.backend.base import ThreadContext, BackendInterfaceBase
import torch
from ktransformers.server.backend.interfaces.transformers import (
    ConfigArgs,
    default_args,
    TextStreamer,
)
from ktransformers.server.schemas.base import ObjectID
from ktransformers.server.config.log import logger
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.custom_modeling_deepseek_v3 import KDeepseekV3ForCausalLM # Specific model import
from ktransformers.models.custom_modeling_deepseek_v2 import KDeepseekV2ForCausalLM # Specific model import
from ktransformers.server.balance_serve.inference.model_runner import ModelRunner
from ktransformers.server.balance_serve.inference.sampling.sampler import Sampler, SamplingOptions
from ktransformers.server.balance_serve.inference.query_manager import QueryManager
from ktransformers.server.balance_serve.inference.forward_batch import ForwardBatchInput, ForwardBatchOutput
from ktransformers.server.balance_serve.sched_rpc import SchedulerClient
from ktransformers.server.balance_serve.settings import sched_ext
from torch.multiprocessing import Queue
import torch.multiprocessing as mp
from ktransformers.server.schemas.endpoints.chat import RawUsage
from ktransformers.server.utils.multi_timer import Profiler
import zmq
import time
import queue
import tempfile
import asyncio
import threading
import hashlib # Added for hashing
from collections import OrderedDict, defaultdict # Added defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import os

# Define rule paths (assuming these exist)
ktransformer_rules_dir = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "./optimize/optimize_rules/")
)
default_optimize_rules = {
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat-serve.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct-serve.yaml",
}

# Helper function to yield tokens from an asyncio queue
async def chat_stream(queue: asyncio.Queue, tokenizer: AutoTokenizer):
    """Asynchronously yields decoded text from tokens received via a queue."""
    streamer = TextStreamer(tokenizer)
    while True:
        token = await queue.get()
        if token is None: # End signal
            s = streamer.end()
            if s:
                yield s
            break
        decoded_text = streamer.put(token)
        if decoded_text:
            yield decoded_text # Yield only non-empty strings

# Helper function to update query states after generation
def fill_generated_tokens(query_updates: list[sched_ext.QueryUpdate], generated_tokens: torch.Tensor, query_manager: QueryManager = None):
    """Fills generated tokens into QueryUpdate objects and updates QueryManager state."""
    # Ensure generated_tokens is on CPU for item() access if necessary
    generated_tokens_cpu = generated_tokens.cpu()
    for i in range(generated_tokens_cpu.size(0)):
        token_item = generated_tokens_cpu[i].item()
        # logger.debug(f"Filling token {token_item} for query {query_updates[i].id}")
        query_updates[i].generated_token = token_item
        query_info = query_manager.query_map.get(query_updates[i].id)
        if query_info and not query_info.is_prefill:
            pos = query_updates[i].active_position
            # Ensure position is within bounds
            if 0 <= pos < query_info.query_tokens.size(0):
                # Ensure token tensor is on the correct device before assignment
                query_info.query_tokens[pos] = generated_tokens[i].to(query_info.query_tokens.device)
            # else:
            # logger.warning(f"Position {pos} out of bounds for query {query_updates[i].id} tokens (size {query_info.query_tokens.size(0)})")

# Helper function to log performance metrics
def report_last_time_performance(profiler: Profiler):
    """Logs performance metrics from the Profiler."""
    try:
        tokenize_time = profiler.get_timer_sec('tokenize')
        prefill_time = profiler.get_timer_sec('prefill')
        decode_time = profiler.get_timer_sec('decode')
        prefill_count = profiler.get_counter('prefill')
        decode_count = profiler.get_counter('decode')

        prefill_tps = prefill_count / prefill_time if prefill_time > 0 else 0
        decode_tps = decode_count / decode_time if decode_time > 0 else 0

        logger.info(f'Performance(T/s): prefill {prefill_tps:.2f}, decode {decode_tps:.2f}. Time(s): tokenize {tokenize_time:.3f}, prefill {prefill_time:.3f}, decode {decode_time:.3f}')
    except Exception as e:
        logger.warning(f'Performance statistics not recorded or error during reporting: {e}')

# Engine class running in a separate process
class Engine:
    """Handles the core inference loop, model loading, and communication with the scheduler."""
    sched_client : SchedulerClient
    updates : list[sched_ext.QueryUpdate]
    batch : sched_ext.BatchQueryTodo
    model_runner: ModelRunner
    sampler: Sampler
    query_manager: QueryManager
    cache: KDeepSeekV3Cache # Assuming specific cache type

    def __init__(self, args: ConfigArgs = default_args, generated_token_queue:Queue = None, broadcast_endpoint: str = None):
        self.args = args

        # Apply args to Config singleton (necessary in separate process)
        cfg = Config()
        for key, value in vars(args).items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)

        self.device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Engine initializing on device: {self.device}")

        self.sched_client = SchedulerClient(args.sched_port)
        self.updates = []
        config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)

        # Initialize cache based on model type - assuming KDeepSeekV3Cache for now
        # TODO: Make cache type dynamic based on config.architectures[0]
        if hasattr(config, 'num_hidden_layers') and hasattr(config, 'max_position_embeddings'):
            # Estimate cache size if needed, replace with actual logic if KDeepSeekV3Cache requires it
            # Example estimation, adjust as necessary
            # estimated_cache_size = config.num_hidden_layers * config.max_position_embeddings * config.hidden_size * 2 # Example
            self.cache = KDeepSeekV3Cache(config, self.args.page_size) # Pass necessary args
        else:
            logger.error("Could not determine necessary parameters for KDeepSeekV3Cache from config.")
            raise ValueError("Failed to initialize cache due to missing config parameters.")


        self.gen_queue = generated_token_queue

        logger.info(f"Getting inference context from sched_client.")
        try:
            inference_context_raw = self.sched_client.get_inference_context_raw()
            logger.info(f"Got raw inference context, rebuilding.")
            inference_context = self.sched_client.rebuild_inferece_context(inference_context_raw)
            logger.info(f"Rebuilt inference context, loading into cache.")
            # Ensure cache is loaded correctly
            if hasattr(self.cache, 'load'):
                self.cache.load(inference_context)
                logger.info(f"KV cache loaded successfully.")
                # Assuming k_cache gives info about block numbers
                if hasattr(inference_context, 'k_cache') and inference_context.k_cache:
                    self.block_num = inference_context.k_cache[0].size(1) # Example access
                else:
                    logger.warning("Could not determine block_num from inference_context.")
                    self.block_num = args.cache_lens // args.page_size # Fallback estimate
            else:
                logger.error("Cache object does not have a 'load' method.")
                raise AttributeError("Cache object missing 'load' method")

        except Exception as e:
            logger.error(f"Failed to get or load inference context: {e}", exc_info=True)
            raise

        # Load model architecture based on config
        with torch.device("meta"): # Load on meta device first
            if config.architectures[0] == "DeepseekV3ForCausalLM":
                self.model = KDeepseekV3ForCausalLM(config, self.cache)
            elif config.architectures[0] == "DeepseekV2ForCausalLM":
                self.model = KDeepseekV2ForCausalLM(config, self.cache)
            # Add other model types here if needed
            # elif config.architectures[0] == "AnotherModelType":
            #     self.model = KAnotherModelType(config, self.cache)
            else:
                logger.error(f"Unsupported model architecture: {config.architectures[0]}")
                raise NotImplementedError(f"Model architecture {config.architectures[0]} not supported")

        # Initialize ZMQ publisher for broadcasting batches (optional)
        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        try:
            self.pub_socket.bind(f"ipc://{broadcast_endpoint}")
            logger.info(f"Publisher bound to ipc://{broadcast_endpoint}")
        except zmq.ZMQError as e:
            logger.error(f"Failed to bind publisher socket: {e}", exc_info=True)
            # Decide if this is critical or can be ignored
            # raise # Or handle gracefully

        # Load GenerationConfig
        try:
            generation_config = GenerationConfig.from_pretrained(args.model_dir)
        except OSError:
            logger.warning(f"GenerationConfig not found in {args.model_dir}, using defaults.")
            generation_config = GenerationConfig(
                max_length=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True # Default to sampling
            )

        # Determine optimization rules path
        if args.optimize_config_path:
            optimize_config_path = args.optimize_config_path
        elif config.architectures[0] in default_optimize_rules:
            optimize_config_path = default_optimize_rules[config.architectures[0]]
            logger.info(f"Using default optimization rule: {optimize_config_path}")
        else:
            logger.warning(f"No optimization rule found for {config.architectures[0]}.")
            optimize_config_path = None # No optimization

        # Get GGUF path
        gguf_path = args.gguf_path
        if gguf_path is None and optimize_config_path:
            # Only prompt if optimization rules exist, otherwise loading might fail
            gguf_path = input(
                "Please input the path of your GGUF file (required for optimization): "
            )
            if not gguf_path:
                logger.error("GGUF path is required for optimization but not provided.")
                raise ValueError("GGUF path needed for optimization.")

        # Optimize and load weights if paths are valid
        if optimize_config_path and gguf_path:
            try:
                logger.info(f"Optimizing and loading GGUF from: {gguf_path} with rules: {optimize_config_path}")
                optimize_and_load_gguf(self.model, optimize_config_path, gguf_path, config)
                logger.info("Model optimized and loaded from GGUF.")
            except Exception as e:
                logger.error(f"Failed during optimize_and_load_gguf: {e}", exc_info=True)
                raise
        else:
            logger.warning("Skipping GGUF optimization/loading (missing paths or rules). Model might not be fully loaded.")
            # Potentially add logic here to load from another format if GGUF isn't used


        self.model.generation_config = generation_config
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

        self.model.eval() # Set to evaluation mode

        # Initialize model wrapper (assuming it handles device placement)
        try:
            # Ensure block_num is valid before initializing wrapper
            if not isinstance(self.block_num, int) or self.block_num <= 0:
                logger.error(f"Invalid block_num: {self.block_num}. Cannot initialize model wrapper.")
                raise ValueError("Invalid block_num for model wrapper initialization.")

            self.model.init_wrapper(self.args.use_cuda_graph, self.device, args.max_batch_size, self.block_num)
            logger.info("Model wrapper initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize model wrapper: {e}", exc_info=True)
            raise

        # Initialize runner and sampler
        self.model_runner = ModelRunner(self.model, self.device, self.args.use_cuda_graph, page_size = args.page_size)
        self.sampler = Sampler()
        self.query_manager = QueryManager(device = self.device, page_size = args.page_size)


    # Corrected: Added Tuple type hint import
    def sampling(self, forward_output: ForwardBatchOutput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs sampling on the model output logits."""
        # Ensure logits are on the correct device
        logits = forward_output.logits[0].to(self.device) # Assuming single batch output for now

        # Prepare sampling options
        temperatures = forward_output.temperatures[0].to(self.device) if hasattr(forward_output, "temperatures") and forward_output.temperatures else None
        top_ps = forward_output.top_ps[0].to(self.device) if hasattr(forward_output, "top_ps") and forward_output.top_ps else None

        # Create SamplingOptions instance
        # Ensure bsz matches logits batch size
        bsz = logits.size(0)
        sample_options = SamplingOptions(
            bsz=bsz,
            device=self.device,
            pretrained_config=self.model.generation_config,
            temperatures=temperatures,
            top_ps=top_ps
        )

        # Perform sampling
        generated_tokens, probs = self.sampler(logits, sample_options)
        return generated_tokens, probs

    def loop(self):
        """The main inference loop."""
        logger.info("Engine loop started.")
        next_batch = None

        while True:
            try:
                # 1. Run model if there's a batch from the previous iteration
                self.batch = next_batch
                if self.batch is not None:
                    # logger.debug(f"Running model for batch with {len(self.batch.query_ids)} queries.")
                    self.model_runner.run(self.batch, self.query_manager)
                    # logger.debug("Model run initiated.")

                # 2. Send generated tokens from the *completed* previous batch to the queue
                if len(self.updates) > 0:
                    # logger.debug(f"Sending {len(self.updates)} updates to token queue.")
                    for q_update in self.updates:
                        if q_update.is_prefill:
                            continue # Don't send tokens during prefill

                        token_to_send = q_update.generated_token if not q_update.decode_done else None
                        # logger.debug(f"Putting token {token_to_send} for query {q_update.id} into queue.")
                        try:
                            # Use put with timeout to prevent indefinite blocking
                            self.gen_queue.put((q_update.id, token_to_send), timeout=1.0)
                        except queue.Full:
                            logger.warning(f"Token queue full for query {q_update.id}, token {token_to_send} might be lost.")
                        except Exception as e:
                            logger.error(f"Error putting token into queue: {e}", exc_info=True)


                # 3. Get the next batch from the scheduler, sending the results of the previous one
                # logger.debug("Requesting next batch from scheduler.")
                next_batch = self.sched_client.update_last_batch(self.updates)
                # logger.debug(f"Received next batch. Query IDs: {next_batch.query_ids if next_batch else 'None'}")

                # Reset next_batch if it's empty
                if next_batch is not None and not next_batch.query_ids:
                    # logger.debug("Received empty batch, setting next_batch to None.")
                    next_batch = None

                # 4. Broadcast the next batch (optional)
                if hasattr(self, 'pub_socket') and self.pub_socket and not self.pub_socket.closed:
                    try:
                        # logger.debug("Broadcasting next batch.")
                        self.pub_socket.send_pyobj(next_batch)
                    except Exception as e:
                        logger.error(f"Error broadcasting batch: {e}", exc_info=True)


                # 5. Add new queries from the next batch to the query manager
                if next_batch is not None:
                    # logger.debug(f"Adding {len(next_batch.query_ids)} queries from next batch to query manager.")
                    self.query_manager.add_query(next_batch)

                # 6. Process the results of the *completed* model run (if any)
                if self.batch is not None:
                    # logger.debug("Syncing model runner.")
                    self.model_runner.sync() # Wait for GPU operations to complete
                    # logger.debug(f"Model execution time (GPU): {self.model_runner.model_time:.3f} ms")

                    # Perform sampling
                    # logger.debug("Performing sampling.")
                    generated_tokens, probs = self.sampling(self.model_runner.output)
                    # logger.debug(f"Sampling complete. Generated tokens shape: {generated_tokens.shape}")

                    # Update query states based on the completed batch
                    # logger.debug("Updating query manager state.")
                    self.updates = self.query_manager.update(self.batch)
                    # logger.debug(f"Generated {len(self.updates)} query updates.")

                    # Fill the generated tokens into the updates
                    # logger.debug("Filling generated tokens into updates.")
                    fill_generated_tokens(self.updates, generated_tokens, self.query_manager)
                else:
                    # No batch was run in this iteration
                    # logger.debug("No batch run in this iteration, clearing updates.")
                    self.updates = []

                # Small sleep if no work was done to prevent high CPU usage
                if self.batch is None and next_batch is None:
                    # logger.debug("No batch processed or received, sleeping.")
                    time.sleep(0.01)

            except ConnectionRefusedError:
                logger.error("Connection to scheduler refused. Is the scheduler running?")
                time.sleep(5) # Wait before retrying
            except Exception as e:
                logger.error(f"Error in engine loop: {e}", exc_info=True)
                # Decide how to handle errors (e.g., retry, exit)
                time.sleep(1) # Avoid rapid error loops
                self.updates = [] # Clear updates on error
                next_batch = None # Clear next batch on error


# Wrapper class for thread context specific to BalanceServe
class BalanceServeThreadContext(ThreadContext):
    """Manages conversation context for the BalanceServe backend."""
    def get_local_messages(self):
        """Formats messages for the model input."""
        local_messages = []
        for m in self.messages:
            # Ensure content is text before getting it
            content = m.get_text_content() if hasattr(m, 'get_text_content') else str(m.content)
            local_messages.append({"role": m.role.value, "content": content})
        return local_messages

# Target function for the engine process
def run_engine(args, token_queue, broadcast_endpoint, start_event):
    """Initializes and runs the Engine loop."""
    try:
        logger.info("Engine process starting...")
        engine = Engine(args, token_queue, broadcast_endpoint)
        if args.use_cuda_graph:
            logger.info("Warming up CUDA graph...")
            engine.model_runner.warmup()
            logger.info("CUDA graph warmup complete.")
        start_event.set() # Signal that initialization is complete
        engine.loop()
    except Exception as e:
        logger.critical(f"Engine process failed: {e}", exc_info=True)
        start_event.set() # Signal completion even on error to unblock main process
        # Consider more robust error handling/reporting here
        raise # Re-raise exception to make the process exit non-zero


# Main interface class for the BalanceServe backend
class BalanceServeInterface(BackendInterfaceBase):
    """Interface for interacting with the BalanceServe backend engine."""
    use_static_cache: bool = False # BalanceServe manages its own cache via scheduler

    # Type hints for attributes
    args: ConfigArgs
    tokenizer: AutoTokenizer
    sched_client: SchedulerClient
    streamer: TextStreamer
    token_queue: Queue # MP Queue from Engine
    active_prefills: Dict[str, asyncio.Future] # Hash -> Future[query_id]
    query_token_queues: Dict[str, List[asyncio.Queue]] # query_id -> List[consumer_queues]
    thread_map: Dict[str, str] # thread_id -> query_id
    proxy_lock: asyncio.Lock

    def __init__(self, args: ConfigArgs = default_args):
        self.args = args
        self.thread_map = {} # Maps thread_id to query_id
        processes = []
        # Create a temporary file for IPC endpoint, ensure it's cleaned up
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_f:
                self.broadcast_endpoint = tmp_f.name
            logger.info(f"Using temporary IPC endpoint: {self.broadcast_endpoint}")
        except Exception as e:
            logger.error(f"Failed to create temporary file for IPC: {e}", exc_info=True)
            # Fallback or raise error
            self.broadcast_endpoint = f"/tmp/ktransformers_ipc_{os.getpid()}" # Example fallback
            logger.warning(f"Falling back to IPC endpoint: {self.broadcast_endpoint}")


        ctx = mp.get_context("spawn")
        self.token_queue = ctx.Queue(maxsize=10000) # Increased queue size
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
        self.sched_client = SchedulerClient(args.sched_port)
        self.streamer = TextStreamer(self.tokenizer)

        # --- Attributes for prefill caching ---
        self.active_prefills = {}
        self.query_token_queues = defaultdict(list) # Maps query_id to list of consumer queues
        self.proxy_lock = asyncio.Lock() # To protect access to shared dictionaries
        # ---------------------------------

        # --- Start Engine Process ---
        start_event = ctx.Event()
        p = ctx.Process(target=run_engine, args=(self.args, self.token_queue, self.broadcast_endpoint, start_event))
        p.start()
        processes.append(p)
        logger.info("Waiting for engine process to initialize...")
        start_event.wait(timeout=60.0) # Add timeout
        if not start_event.is_set():
            logger.error("Engine process initialization timed out.")
            if p.is_alive():
                p.terminate() # Terminate if stuck
            raise TimeoutError("Engine process failed to initialize within timeout.")

        # Check if process started correctly
        if not p.is_alive():
            logger.error("Engine process failed to start or terminated prematurely.")
            # Clean up temporary file if process failed
            if self.broadcast_endpoint and os.path.exists(self.broadcast_endpoint) and self.broadcast_endpoint.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(self.broadcast_endpoint)
                    logger.info(f"Cleaned up temporary IPC file: {self.broadcast_endpoint}")
                except OSError as unlink_e:
                    logger.error(f"Error cleaning up IPC file {self.broadcast_endpoint}: {unlink_e}")
            raise RuntimeError("Engine process failed to start.")
        logger.info("Engine process started successfully.")
        self._engine_process = p # Store process handle for potential cleanup

    def __del__(self):
        # Cleanup resources on object deletion
        logger.info("BalanceServeInterface shutting down...")
        # Close scheduler client connection
        if hasattr(self, 'sched_client') and self.sched_client:
            # Assuming SchedulerClient has a close method or similar
            try:
                # self.sched_client.close() # Add a close method if needed
                pass
            except Exception as e:
                logger.error(f"Error closing scheduler client: {e}")

        # Terminate engine process if still alive
        if hasattr(self, '_engine_process') and self._engine_process and self._engine_process.is_alive():
            logger.info("Terminating engine process...")
            self._engine_process.terminate()
            self._engine_process.join(timeout=5.0) # Wait briefly for termination
            if self._engine_process.is_alive():
                logger.warning("Engine process did not terminate gracefully, killing.")
                self._engine_process.kill()

        # Clean up temporary IPC file
        if hasattr(self, 'broadcast_endpoint') and self.broadcast_endpoint and os.path.exists(self.broadcast_endpoint) and self.broadcast_endpoint.startswith(tempfile.gettempdir()):
            try:
                os.unlink(self.broadcast_endpoint)
                logger.info(f"Cleaned up temporary IPC file: {self.broadcast_endpoint}")
            except OSError as e:
                logger.error(f"Error cleaning up IPC file {self.broadcast_endpoint}: {e}")


    def format_and_tokenize_input_ids(self, thread_id: ObjectID, messages: List) -> torch.Tensor:
        """Formats messages and tokenizes them, returning a CPU tensor."""
        # Role conversion and merging adjacent user messages
        for m in messages:
            if m["role"] == "system":
                # logger.warning(f'Changing role from system to user for message in thread {thread_id}')
                m["role"] = "user"

        new_messages = [messages[0]] if messages else []
        for m in messages[1:]:
            if m["role"] == "user" and new_messages and new_messages[-1]["role"] == "user":
                # logger.warning(f"Merging adjacent user messages in thread {thread_id}")
                new_messages[-1]["content"] += '\n' + m["content"]
            elif m["role"] in ["user", "assistant"]: # Only include user/assistant roles
                new_messages.append(m)
            # else:
            # logger.debug(f"Skipping message with role {m.get('role')} in thread {thread_id}")


        if not new_messages:
            logger.warning(f"No valid user/assistant messages found for thread {thread_id} after filtering.")
            return torch.tensor([[]], dtype=torch.long) # Return empty tensor for empty input

        # Apply chat template
        try:
            input_str: str = self.tokenizer.apply_chat_template(
                new_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.error(f"Error applying chat template for thread {thread_id}: {e}", exc_info=True)
            # Fallback or re-raise
            input_str = "\n".join([f"{m['role']}: {m['content']}" for m in new_messages]) + "\nassistant:"


        # Remove <think> token if present at the end
        if input_str.endswith('<think>\n'):
            input_str = input_str[:-len('<think>\n')]

        # Encode the string
        try:
            input_ids = self.tokenizer.encode(input_str, return_tensors="pt")
        except Exception as e:
            logger.error(f"Error encoding input string for thread {thread_id}: {e}", exc_info=True)
            return torch.tensor([[]], dtype=torch.long) # Return empty on encoding error

        # logger.debug(f"Formatted and tokenized input for thread {thread_id}. Shape: {input_ids.shape}")
        return input_ids.cpu() # Return CPU tensor

    def tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenizes a raw prompt string, returning a CPU tensor."""
        return self.tokenizer.encode(prompt, return_tensors="pt").cpu()

    async def queue_proxy(self):
        """Forwards tokens from the main engine queue to specific consumer queues."""
        logger.info("Queue Proxy Started")
        while True:
            try:
                # Use blocking get with a timeout
                query_id, token = self.token_queue.get(timeout=0.1)

                async with self.proxy_lock:
                    if query_id in self.query_token_queues:
                        # logger.debug(f"Proxy: Got token {token} for query {query_id}. Distributing to {len(self.query_token_queues[query_id])} queues.")
                        queues_to_remove = []
                        for i, q in enumerate(self.query_token_queues[query_id]):
                            try:
                                q.put_nowait(token)
                            except asyncio.QueueFull:
                                logger.warning(f"Proxy: Consumer queue {i} full for query {query_id}. Token might be dropped.")
                            except Exception as e: # Catch potential errors during put_nowait
                                logger.error(f"Proxy: Error putting token to consumer queue {i} for query {query_id}: {e}")
                                # Optionally mark queue for removal if it causes persistent errors
                                # queues_to_remove.append(q)


                        # Remove queues that caused errors (optional)
                        # for q_rem in queues_to_remove:
                        #    try:
                        #        self.query_token_queues[query_id].remove(q_rem)
                        #    except ValueError: pass # Already removed

                        if token is None: # End signal
                            # logger.debug(f"Proxy: End signal received for query {query_id}. Cleaning up.")
                            # Clear the list associated with the query_id
                            self.query_token_queues.pop(query_id, None)
                    # else:
                    # logger.debug(f"Proxy: Received token for inactive/unknown query {query_id}. Discarding.")

            except queue.Empty:
                await asyncio.sleep(0.01) # Yield control when queue is empty
            except Exception as e:
                # Catch potential exceptions if the queue is closed during shutdown
                if isinstance(e, (EOFError, BrokenPipeError)):
                    logger.warning(f"Queue proxy encountered EOF/BrokenPipe, likely during shutdown: {e}")
                    break # Exit loop on shutdown signal
                logger.error(f"Error in queue_proxy: {e}", exc_info=True)
                await asyncio.sleep(1) # Avoid rapid error loops

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Manages the queue proxy task using FastAPI's lifespan events."""
        proxy_task = asyncio.create_task(self.queue_proxy())
        logger.info("Queue proxy task started via lifespan manager.")
        yield
        # Cleanup on shutdown
        logger.info("Application shutting down. Cancelling proxy task...")
        proxy_task.cancel()
        try:
            await proxy_task
        except asyncio.CancelledError:
            logger.info("Queue proxy task successfully cancelled.")
        except Exception as e:
            logger.error(f"Error during proxy task cancellation: {e}", exc_info=True)


    async def _get_token_stream(self, query_id: str) -> AsyncIterable[int]:
        """Creates and manages a consumer queue for a specific query ID."""
        my_queue = asyncio.Queue()
        async with self.proxy_lock:
            self.query_token_queues[query_id].append(my_queue)
            # logger.debug(f"Stream: Consumer added for query {query_id}. Total consumers: {len(self.query_token_queues[query_id])}")

        try:
            while True:
                token = await my_queue.get()
                # logger.debug(f"Stream: Received token {token} for query {query_id}")
                if token is None: # End signal
                    # logger.debug(f"Stream: End signal received for query {query_id}")
                    break
                yield token
                my_queue.task_done()
        finally:
            # logger.debug(f"Stream: Cleaning up consumer queue for query {query_id}")
            async with self.proxy_lock:
                if query_id in self.query_token_queues:
                    try:
                        self.query_token_queues[query_id].remove(my_queue)
                        # logger.debug(f"Stream: Consumer removed for query {query_id}. Remaining: {len(self.query_token_queues[query_id])}")
                        # Let the proxy handle removing the entry when the list is empty upon receiving None
                    except ValueError:
                        # logger.warning(f"Stream: Queue already removed for query {query_id}, possibly by proxy.")
                        pass # Queue already removed, ignore


    async def inference(self, local_messages, thread_id: str, temperature: Optional[float] = None, top_p: Optional[float] = None) -> AsyncIterable[Tuple[str, Optional[str]] | RawUsage]:
        """Handles inference requests, utilizing prefill caching."""
        # 1. Calculate Hash
        m = hashlib.sha256()
        m.update(str(local_messages).encode())
        # Include relevant generation parameters in the hash
        temp_to_hash = temperature if temperature is not None else self.args.temperature
        top_p_to_hash = top_p if top_p is not None else self.args.top_p
        m.update(str(temp_to_hash).encode())
        m.update(str(top_p_to_hash).encode())
        # Consider adding other params like max_new_tokens if they affect prefill state
        prompt_hash = m.hexdigest()

        query_id_to_use = None
        is_original_request = False
        query_id_future = None

        # 2. Check Cache / Start New Prefill (Atomic Check)
        async with self.proxy_lock:
            if prompt_hash in self.active_prefills:
                # logger.info(f"Prefill Cache HIT for hash {prompt_hash[:8]}...")
                query_id_future = self.active_prefills[prompt_hash]
            else:
                # logger.info(f"Prefill Cache MISS for hash {prompt_hash[:8]}. Starting new prefill.")
                is_original_request = True
                query_id_future = asyncio.Future()
                self.active_prefills[prompt_hash] = query_id_future

        # 3. Process Request
        if is_original_request:
            # --- This is the first request for this prompt hash ---
            profiler_orig = Profiler() # Profiler for the original request
            actual_query_id = None # Initialize to None
            try:
                # --- Tokenize ---
                profiler_orig.create_and_start_timer("tokenize")
                if isinstance(local_messages, List):
                    input_ids = self.format_and_tokenize_input_ids(thread_id, local_messages)
                elif isinstance(local_messages, str):
                    input_ids = self.tokenize_prompt(local_messages)
                else:
                    raise ValueError("local_messages should be List or str")

                if input_ids.numel() == 0:
                    raise ValueError("Tokenization resulted in empty input.")


                # --- Force Think Token (if enabled) ---
                if Config().user_force_think:
                    input_ids = input_ids.cpu() # Ensure on CPU
                    think_tokens = self.tokenizer.encode("<think>\n", add_special_tokens=False)
                    token_thinks = torch.tensor([think_tokens], dtype=torch.long)
                    input_ids = torch.cat([input_ids, token_thinks], dim=1)

                profiler_orig.pause_timer("tokenize")
                profiler_orig.create_and_start_timer("prefill")

                # --- Prepare QueryAdd ---
                query_add = sched_ext.QueryAdd()
                query_add.query_token = input_ids[0].cpu().tolist() # Send list of ints
                query_length = len(query_add.query_token)
                query_add.query_length = query_length
                profiler_orig.set_counter("prefill", query_length)

                # Define stop criteria (ensure tokens are ints)
                stop_criteria_tokens = [
                    self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False) if self.tokenizer.eos_token else [],
                    self.tokenizer.encode("<|im_end|>", add_special_tokens=False) # Add other common stop tokens if needed
                ]
                # Filter out empty lists if eos_token is None
                query_add.stop_criteria = [tokens for tokens in stop_criteria_tokens if tokens]


                # Set sampling parameters, ensuring they are not zero
                temp = temperature if temperature is not None else self.args.temperature
                tp = top_p if top_p is not None else self.args.top_p
                query_add.sample_options.temperature = max(0.0001, temp)
                query_add.sample_options.top_p = max(0.0001, tp)

                # Estimate length, ensuring it's valid
                query_add.estimated_length = min(self.args.cache_lens, query_length + self.args.max_new_tokens)
                if query_add.estimated_length <= query_add.query_length:
                    # Adjust if max_new_tokens is too small or zero
                    query_add.estimated_length = query_length + 1 # Ensure at least one token can be generated
                    logger.warning(f"Adjusted estimated_length for query {thread_id} as initial estimate was too small.")
                    # Alternatively, raise error if query_length already exceeds cache_lens
                    if query_length >= self.args.cache_lens:
                        raise ValueError(f"Query length ({query_length}) exceeds cache length ({self.args.cache_lens}).")


                # --- Add Query to Scheduler ---
                actual_query_id = self.sched_client.add_query(query_add)
                # logger.info(f"New query added to scheduler with ID: {actual_query_id} for hash {prompt_hash[:8]}")
                self.thread_map[thread_id] = actual_query_id
                query_id_to_use = actual_query_id

                # Signal waiting tasks by setting the Future's result
                query_id_future.set_result(actual_query_id)

                # --- Start Streaming Results ---
                if Config().user_force_think:
                    yield '<think>\n', None # Yield think token immediately

                self.streamer.reset()
                token_count = 0
                is_first_token_yielded = False # Track if the first *decoded* token was yielded
                async for token_id in self._get_token_stream(actual_query_id):
                    # Prefill ends when the first token arrives from the engine
                    if not is_first_token_yielded:
                        profiler_orig.pause_timer("prefill")
                        profiler_orig.create_and_start_timer("decode")
                        profiler_orig.set_counter("decode", 0)

                    # Decode and yield text
                    decoded_text = self.streamer.put(token_id)
                    if decoded_text:
                        yield decoded_text, None
                        is_first_token_yielded = True # Mark first yield

                    # Increment decode counter *after* the first token arrives
                    if not is_first_token_yielded:
                        pass # Still in prefill phase according to profiler
                    else:
                        profiler_orig.inc("decode")
                    token_count += 1


                # --- Handle Final Token & Finish Reason ---
                final_text = self.streamer.end()
                if final_text:
                    yield final_text, None

                # Pause decode timer after loop finishes
                # Handle case where no tokens were generated (decode timer never started)
                if not is_first_token_yielded and profiler_orig.timers.get('decode') is None:
                    # If prefill finished but no decode tokens arrived (e.g., immediate stop)
                    profiler_orig.pause_timer("prefill") # Ensure prefill timer is stopped
                    profiler_orig.create_timer("decode") # Create decode timer
                    profiler_orig.set_counter("decode", 0) # Set count to 0
                profiler_orig.pause_timer("decode") # Pause decode timer

                report_last_time_performance(profiler_orig) # Report performance

                # Determine finish reason
                decode_count = profiler_orig.get_counter('decode')
                # Calculate max_new based on available cache space
                max_new = max(0, self.args.cache_lens - query_length - 1)
                max_new = min(max_new, self.args.max_new_tokens) # Also respect max_new_tokens limit

                # logger.debug(f"Query {actual_query_id}: Decoded {decode_count} tokens. Max allowed: {max_new}. Query length: {query_length}. Cache: {self.args.cache_lens}")
                finish_reason = "length" if decode_count >= max_new else "stop"
                # logger.info(f"Original request {actual_query_id} finished with reason: {finish_reason}")
                yield "", finish_reason

                # --- Yield Usage ---
                usage = RawUsage(
                    tokenize_time=profiler_orig.get_timer_sec('tokenize'),
                    prefill_time=profiler_orig.get_timer_sec('prefill'),
                    decode_time=profiler_orig.get_timer_sec('decode'),
                    prefill_count=profiler_orig.get_counter('prefill'),
                    decode_count=profiler_orig.get_counter('decode'),
                )
                yield usage

            except Exception as e:
                logger.error(f"Error processing original request for hash {prompt_hash[:8]}: {e}", exc_info=True)
                # Ensure future is set with exception if it wasn't set with query_id
                if not query_id_future.done():
                    query_id_future.set_exception(e)
                # Clean up token queue if query was added but streaming failed
                if actual_query_id:
                    async with self.proxy_lock:
                        self.query_token_queues.pop(actual_query_id, None)
                raise # Re-raise the exception for the caller
            finally:
                # Clean up active prefill entry *always*
                async with self.proxy_lock:
                    removed_future = self.active_prefills.pop(prompt_hash, None)
                    # logger.debug(f"Cleaned up active_prefills for hash {prompt_hash[:8]}. Future removed: {removed_future is not None}")

        else:
            # --- This request joined an existing prefill ---
            try:
                # Wait for the original request to get the query_id
                query_id_to_use = await query_id_future
                # logger.info(f"Joined existing prefill for query ID: {query_id_to_use}")

                # Stream results from the shared queue
                self.streamer.reset()
                is_first_token = True # For yielding think token
                async for token_id in self._get_token_stream(query_id_to_use):
                    if is_first_token and Config().user_force_think:
                        yield '<think>\n', None
                        is_first_token = False

                    decoded_text = self.streamer.put(token_id)
                    if decoded_text:
                        yield decoded_text, None

                final_text = self.streamer.end()
                if final_text:
                    yield final_text, None

                # logger.info(f"Joined request for query {query_id_to_use} finished streaming.")
                # Yield default/unknown finish reason and no usage for joined requests
                yield "", "stop" # Assume stop, as we don't know the original's reason
                yield None # No usage info

            except asyncio.CancelledError:
                logger.info(f"Joined request for hash {prompt_hash[:8]} cancelled.")
                yield "", "cancelled" # Indicate cancellation if possible
                yield None
            except Exception as e:
                logger.error(f"Error processing joined request for hash {prompt_hash[:8]} (Future exception: {query_id_future.exception()}): {e}", exc_info=True)
                # Yield minimal info to signal error
                yield "", "failed" # Indicate failure
                yield None
