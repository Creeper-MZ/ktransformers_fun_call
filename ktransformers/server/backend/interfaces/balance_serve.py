from typing import Any, AsyncIterator, List, Optional, Set
from ktransformers.models.custom_cache import KDeepSeekV3Cache
from transformers import (
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
    StaticCache,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from ktransformers.server.config.config import Config
from ..base import ThreadContext, BackendInterfaceBase
import torch
from ktransformers.server.backend.interfaces.transformers import (
    ConfigArgs,
    default_args,
    TextStreamer,
)
from ktransformers.server.schemas.base import ObjectID
from ktransformers.server.config.log import logger
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.custom_modeling_deepseek_v3 import KDeepseekV3ForCausalLM
from ktransformers.models.custom_modeling_deepseek_v2 import KDeepseekV2ForCausalLM
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
import hashlib
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import os
import torch.multiprocessing as mp


ktransformer_rules_dir = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "./optimize/optimize_rules/")
)
default_optimize_rules = {
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat-serve.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct-serve.yaml",
}

async def chat_stream(queue: asyncio.Queue, tokenizer: AutoTokenizer):
    streamer = TextStreamer(tokenizer)
    while True:
        token = await queue.get()
        #print(f"Got token: {token}")
        if token is None:
            # str = f'{token}\n\n'
            # str = model.tokenizer.decode(token)
            s = streamer.end()
            if s is not None:
                yield s
            break

        # str = model.tokenizer.decode(token)
        yield streamer.put(token)



def fill_generated_tokens(query_updates: list[sched_ext.QueryUpdate], generated_tokens: torch.Tensor, query_manager: QueryManager = None):
    #print(len(query_updates), generated_tokens.size(0), generated_tokens)
    for i in range(generated_tokens.size(0)):
        print(generated_tokens[i].item())
        query_updates[i].generated_token = generated_tokens[i].item()
        if not query_manager.query_map[query_updates[i].id].is_prefill:
            pos = query_updates[i].active_position
            query_manager.query_map[query_updates[i].id].query_tokens[pos] = generated_tokens[i]

def report_last_time_performance(profiler: Profiler):
    try:
        tokenize_time = profiler.get_timer_sec('tokenize')
        prefill_time = profiler.get_timer_sec('prefill')
        decode_time = profiler.get_timer_sec('decode')
        prefill_count = profiler.get_counter('prefill')
        decode_count = profiler.get_counter('decode')

        logger.info(f'Performance(T/s): prefill {prefill_count/prefill_time}, decode {decode_count/decode_time}. Time(s): tokenize {tokenize_time}, prefill {prefill_time}, decode {decode_time}')
    except:
        logger.info(f'Performance statistics not recorded')

class Engine:
    sched_client : SchedulerClient
    updates : list[sched_ext.QueryUpdate]
    batch : sched_ext.BatchQueryTodo
    model_runner: ModelRunner
    sampler: Sampler
    query_manager: QueryManager
    cache: KDeepSeekV3Cache
    def __init__(self, args: ConfigArgs = default_args, generated_token_queue:Queue = None, broadcast_endpoint: str = None):
        self.args = args

        # 子进程和父进程无法共享 config 变量
        for key, value in vars(args).items():
            if value is not None and hasattr(Config(), key):
                setattr(Config(), key, value)

        self.device = self.args.device
        self.sched_client = SchedulerClient(args.sched_port)
        self.updates = []
        config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        self.cache = KDeepSeekV3Cache(config, self.args.page_size)

        self.gen_queue = generated_token_queue

        print(f"Getting inference context from sched_client.")
        inference_context = self.sched_client.get_inference_context_raw()
        print(f"Got inference context, sending it to subscribers.")
        inference_context = self.sched_client.rebuild_inferece_context(inference_context)
        self.cache.load(inference_context)
        print(f"kv_cache loaded successfully.")

        self.block_num = inference_context.k_cache[0].size(1)
        with torch.device("meta"):
            if config.architectures[0] == "DeepseekV3ForCausalLM":
                self.model = KDeepseekV3ForCausalLM(config, self.cache)
            elif config.architectures[0] == "DeepseekV2ForCausalLM":
                self.model = KDeepseekV2ForCausalLM(config, self.cache)
        # print(self.block_num)

        context = zmq.Context()


        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(f"ipc://{broadcast_endpoint}")
        # time.sleep(1) # make sure all subscribers are ready


        try:
            generation_config = GenerationConfig.from_pretrained(args.model_dir)
        except:
            generation_config = GenerationConfig(
                max_length=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True
            )

        if args.optimize_config_path is None:
            optimize_config_path = default_optimize_rules[config.architectures[0]]

        else:
            optimize_config_path = args.optimize_config_path
        gguf_path = args.gguf_path
        if gguf_path is None:
            gguf_path = input(
                "please input the path of your gguf file(gguf file in the dir containing input gguf file must all"
                " belong to current model):"
            )
        optimize_and_load_gguf(self.model, optimize_config_path, gguf_path, config)
        self.model.generation_config = generation_config
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

        self.model.eval()
        #@TODO add config
        self.model.init_wrapper(self.args.use_cuda_graph, self.device, args.max_batch_size, self.block_num)

        self.model_runner = ModelRunner(self.model, self.device, self.args.use_cuda_graph, page_size = args.page_size)
        self.sampler = Sampler()
        self.query_manager = QueryManager(device = self.device, page_size = args.page_size)


    def sampling(self, forward_output: ForwardBatchOutput):
        generated_tokens = torch.empty(0, device=self.device, dtype=torch.int32)
        for i in range(forward_output.num_batchs):
            logit = forward_output.logits[i]
            if hasattr(forward_output, "temperatures"):
                temperatures = forward_output.temperatures[i]
            else:
                temperatures = None

            if hasattr(forward_output, "top_ps"):
                top_ps = forward_output.top_ps[i]
            else:
                top_ps = None

            sample_options = SamplingOptions(logit.size(0), self.device, pretrained_config=self.model.generation_config, temperatures=temperatures, top_ps=top_ps)
            generated_tokens, probs=self.sampler(logit, sample_options)
        return generated_tokens, probs

    def loop(self):

        next_batch = None

        while True:
            self.batch = next_batch
            if self.batch is not None:
                self.model_runner.run(self.batch, self.query_manager)

            if len(self.updates) > 0:
                for q in self.updates:
                    if q.is_prefill == True:
                        continue
                    # print(f"Putting token {q.generated_token} into queue for query id: {q.id}")
                    try:
                        self.gen_queue.put((q.id, q.generated_token if q.decode_done == False else None), timeout=5)
                    except queue.Full:
                        pass#print("Queue is full after timeout; unable to put more items.")

            next_batch = self.sched_client.update_last_batch(self.updates)
            if next_batch.query_ids == []:
                next_batch = None
            self.pub_socket.send_pyobj(next_batch)

            if next_batch is not None:
                self.query_manager.add_query(next_batch)


            if self.batch is not None:
                self.model_runner.sync()
                print(f"Model execution time (GPU): {self.model_runner.model_time:.3f} ms")
                # if self.rank == 0:

                generated_tokens, probs = self.sampling( self.model_runner.output)

                self.updates = self.query_manager.update(self.batch)
                fill_generated_tokens(self.updates, generated_tokens, self.query_manager)
            else:
                self.updates = []

class BalanceServeThreadContext(ThreadContext):
    def get_local_messages(self):
        local_messages = []
        for m in self.messages:
            local_messages.append({"role": m.role.value, "content": m.get_text_content()})

        return local_messages


def run_engine(args, token_queue, broadcast_endpoint, event):
    engine = Engine(args, token_queue, broadcast_endpoint)
    if args.use_cuda_graph:
        engine.model_runner.warmup()

    event.set()
    engine.loop()


class BalanceServeInterface(BackendInterfaceBase):
    use_static_cache: bool = True

    model: Any
    tokenizer: AutoTokenizer

    cache: StaticCache
    generated_ids: torch.Tensor
    seq_length: int

    streamer: TextStreamer

    # thread_related
    last_request_id: Optional[str] = None
    ever_generated_ids: Set[int] = set()

    def __init__(self, args: ConfigArgs = default_args):
        self.args = args

        # Initialize response cache
        self.enable_response_cache = True
        self.max_cache_size = 128000
        self.response_cache = {}  # Map input hash -> generated tokens

        logger.info(f"[CACHE] Response caching {'enabled' if self.enable_response_cache else 'disabled'}, max size: {self.max_cache_size}")

        self.queue_map = {}
        self.thread_map = {}
        processes = []
        self.broadcast_endpoint = tempfile.NamedTemporaryFile(delete=False).name # @TODO add to config
        ctx = mp.get_context("spawn")
        self.token_queue = ctx.Queue(maxsize=1000)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        self.sched_client = SchedulerClient(args.sched_port)
        self.streamer = TextStreamer(self.tokenizer)

        start_event = ctx.Event()

        p = ctx.Process(target=run_engine, args=(self.args, self.token_queue, self.broadcast_endpoint, start_event))
        p.start()
        processes.append(p)
        start_event.wait()

    def log_cache_contents(self):
        """Log detailed information about the current cache contents"""
        logger.info(f"===== CACHE CONTENTS =====")
        logger.info(f"Cache size: {len(self.response_cache)} entries")

        for idx, (hash_key, responses) in enumerate(self.response_cache.items()):
            usage = next((item for item in responses if isinstance(item, RawUsage)), None)
            token_count = len([r for r in responses if not isinstance(r, RawUsage)])

            logger.info(f"Entry {idx+1}: Hash={hash_key[:8]}..., Tokens={token_count}")
            if usage:
                logger.info(f"  Prefill time: {usage.prefill_time:.3f}s, Decode time: {usage.decode_time:.3f}s")
                logger.info(f"  Prefill tokens: {usage.prefill_count}, Decode tokens: {usage.decode_count}")

        logger.info(f"=========================")

    def run_queue_proxy(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.queue_proxy())

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        asyncio.create_task(self.queue_proxy())
        yield

    async def queue_proxy(self):
        print("Queue Proxy Started")
        while True:
            try:
                query_id, token = self.token_queue.get_nowait()
                try:
                    # query id might not be allocated yet
                    self.queue_map[query_id].put_nowait(token)
                    #print(f"Proxy Put token: {token} to queue for query id: {query_id}")
                except asyncio.QueueFull:
                    #print(f"Queue for query id: {query_id} is full, waiting to put: {token}")
                    await self.queue_map[query_id].put(token)

            except queue.Empty:
                # print("no new token")
                # await asyncio.sleep(1)
                await asyncio.sleep(0)

    def tokenize_prompt(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.args.device)
        return input_ids

    def format_and_tokenize_input_ids(self, thread_id: ObjectID, messages: List):
        for m in messages:
            if m["role"] == "system":
                logger.warning(f'change {m["role"]} to user')
                m["role"] = "user"

        new_messages = [messages[0]]
        for m in messages[1:]:
            if m["role"] == "user" and new_messages[-1]["role"] == "user":
                logger.warning("merge two adjacent user messages")
                new_messages[-1]["content"] += '\n' + m["content"]
            else:
                new_messages.append(m)
        input_str: str = self.tokenizer.apply_chat_template(new_messages,tokenize=False,add_generation_prompt=True)
        # drop <think> token in chat template
        if input_str.endswith('<think>\n'):
            input_str = input_str[:-len('<think>\n')]
        input_ids = self.tokenizer.encode(input_str, return_tensors="pt").to(self.args.device)
        logger.debug(f"get input ids of shape {input_ids.shape}")
        return input_ids

    async def inference(self, local_messages, thread_id: str, temperature: Optional[float] = None, top_p: Optional[float] = None):
        # Generate a cache key for this input
        if self.enable_response_cache:
            input_str = str(local_messages)
            if temperature is not None:
                input_str += f"_temp_{temperature}"
            if top_p is not None:
                input_str += f"_top_p_{top_p}"

            input_hash = hashlib.md5(input_str.encode()).hexdigest()

            logger.debug(f"[CACHE] Checking cache for input hash: {input_hash}")
            logger.debug(f"[CACHE] Current cache size: {len(self.response_cache)} entries")

            # Check cache for hit
            cached_response = self.response_cache.get(input_hash)
            logger.debug(self.response_cache)
            if cached_response:
                logger.info(f"[CACHE] HIT! Using cached response for input hash: {input_hash[:8]}...")
                logger.debug(f"[CACHE] Cached response contains {len(cached_response)} tokens")

                cached_prefill_time = next((item.prefill_time for item in cached_response if isinstance(item, RawUsage)), 0)
                logger.info(f"[CACHE] Saved approximately {cached_prefill_time:.3f} seconds of prefill time")

                # Return cached response
                for item in cached_response:
                    if isinstance(item, RawUsage):
                        yield item
                    else:
                        yield item
                return
            else:
                logger.info(f"[CACHE] MISS! No cached response for input hash: {input_hash[:8]}...")

        # No cache hit or caching disabled - proceed with normal inference
        profiler = Profiler()
        profiler.create_and_start_timer("tokenize")

        if isinstance(local_messages, List):
            input_ids = self.format_and_tokenize_input_ids(thread_id, local_messages)
        elif isinstance(local_messages, str):
            #local_messages = local_messages[0]['content']
            input_ids = self.tokenize_prompt(local_messages)
        else:
            raise ValueError("local_messages should be List or str")

        if Config().user_force_think:
            token_thinks = torch.tensor([self.tokenizer.encode("<think>\n",add_special_tokens=False)],device=input_ids.device)
            input_ids = torch.cat(
                [input_ids, token_thinks], dim=1
            )

        profiler.pause_timer("tokenize")
        profiler.create_and_start_timer("prefill")

        # For caching, store all responses
        generated_responses = []

        query_add = sched_ext.QueryAdd()
        query_add.query_token = input_ids[0].tolist()
        query_length = input_ids[0].shape[0]
        query_add.query_length = query_length
        profiler.set_counter("prefill", query_length)

        stop_criteria = [self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False),
                         self.tokenizer.encode("<|im_end|>")]
        query_add.stop_criteria = stop_criteria

        if temperature == 0:
            temperature = 0.0001
        query_add.sample_options.temperature = temperature

        if top_p == 0:
            top_p = 0.0001
        query_add.sample_options.top_p = top_p

        query_add.estimated_length = min(self.args.cache_lens, query_length+self.args.max_new_tokens)

        if query_add.estimated_length < query_add.query_length:
            raise Exception(f'query too long: estimated_length={query_add.estimated_length} < query_length={query_add.query_length}')

        query_id = self.sched_client.add_query(query_add)
        queue = asyncio.Queue(maxsize=self.args.max_new_tokens)
        self.queue_map[query_id] = queue
        self.thread_map[thread_id] = query_id
        is_first_token = True

        async for token in chat_stream(self.queue_map[query_id], self.tokenizer):
            if is_first_token:
                is_first_token = False
                profiler.pause_timer("prefill")
                profiler.create_and_start_timer("decode")
                profiler.set_counter("decode", 0)
                if Config().user_force_think:
                    think = '<think>\n'
                    print(think, end="", flush=True)
                    generated_responses.append((think, None))
                    yield think, None
            else:
                profiler.inc("decode")

            # Save token for caching
            generated_responses.append((token, None))
            yield token, None

        profiler.pause_timer("decode")
        report_last_time_performance(profiler)

        token = self.streamer.end()
        if token:
            generated_responses.append((token, None))
            yield token, None

        if profiler.get_counter('decode') >= self.args.max_new_tokens - 1:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        generated_responses.append(("", finish_reason))
        yield "", finish_reason

        usage = RawUsage(
            tokenize_time = profiler.get_timer_sec('tokenize'),
            prefill_time = profiler.get_timer_sec('prefill'),
            decode_time = profiler.get_timer_sec('decode'),
            prefill_count = profiler.get_counter('prefill'),
            decode_count = profiler.get_counter('decode'),
        )

        generated_responses.append(usage)
        yield usage

        # Add to cache if enabled
        if self.enable_response_cache:
            if len(self.response_cache) >= self.max_cache_size:
                oldest_key = list(self.response_cache.keys())[0]
                logger.debug(f"[CACHE] Cache full, removing oldest entry with hash: {oldest_key[:8]}...")
                del self.response_cache[oldest_key]

            logger.info(f"[CACHE] Storing new response in cache with hash: {input_hash[:8]}...")
            logger.debug(f"[CACHE] Response contains {len(generated_responses)} tokens")
            logger.debug(f"[CACHE] New cache size: {len(self.response_cache) + 1} entries")
            self.response_cache[input_hash] = generated_responses

            # Log periodic cache stats
            if len(self.response_cache) % 5 == 0:
                logger.info(f"[CACHE] Stats: size={len(self.response_cache)}, max={self.max_cache_size}")
                total_tokens = sum(len(resp) for resp in self.response_cache.values())
                logger.info(f"[CACHE] Total cached tokens: {total_tokens}")