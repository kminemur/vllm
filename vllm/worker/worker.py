"""A GPU worker class."""
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
import intel_extension_for_pytorch 
import oneccl_bindings_for_pytorch

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
from vllm.utils import get_gpu_memory, get_xpu_memory


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        self.model_runner = ModelRunner(model_config, parallel_config,
                                        scheduler_config)
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None
        self.cpu_cache = None
        self.xpu_cache = None

    def init_model(self) -> None:
        if self.model_config.device == torch.device('cpu'):
            self.rank = 0
            self.device = torch.device("cpu")
        elif self.model_config.device == torch.device('xpu'):
            self.rank = int(os.environ.get('RANK', 0))
            # world_size = int(os.environ.get("WORLD_SIZE",1))
            # self.distributed_init_method = os.environ["INIT_FILE"]
            self.device = torch.device("xpu")            
        else:
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            # Env vars will be set by Ray.
            self.rank = self.rank if self.rank is not None else int(
                os.getenv("RANK", "-1"))
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{local_rank}")
            if self.rank < 0:
                raise ValueError("Invalid or unspecified rank.")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)

        # Initialize the distributed environment.
        _init_distributed_environment(self.parallel_config, self.rank,
                                      self.distributed_init_method)

        # Initialize the model.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int, int]:
        if self.model_config.device == torch.device('xpu'):
            torch.xpu.empty_cache()
            torch.xpu.reset_peak_memory_stats()
            # Execute a forward pass with dummy inputs to profile the memory usage
            # of the model.
            self.model_runner.profile_run()

            # Calculate the number of blocks that can be allocated with the
            # profiled peak memory.
            torch.xpu.synchronize()
            peak_memory = torch.xpu.max_memory_allocated()
            total_xpu_memory = get_xpu_memory()
            cache_block_size = CacheEngine.get_cache_block_size(
                block_size, self.model_config, self.parallel_config)
            print("peak memory: " + str(peak_memory) + " total memory: " + str(total_xpu_memory) +
                  "cache block size: " + str(cache_block_size))
            
            num_xpu_blocks = int(
                (total_xpu_memory * gpu_memory_utilization - peak_memory) //
                cache_block_size)
            num_cpu_blocks = int(cpu_swap_space // cache_block_size)
            num_xpu_blocks = max(num_xpu_blocks, 0)
            num_cpu_blocks = max(num_cpu_blocks, 0)
            torch.xpu.empty_cache()
            
            return 0, num_cpu_blocks, num_xpu_blocks
            
        if self.model_config.device == torch.device('cpu'):
            cache_block_size = CacheEngine.get_cache_block_size(
                block_size, self.model_config, self.parallel_config)
            num_gpu_blocks = 0
            num_cpu_blocks = int(cpu_swap_space // cache_block_size)
            num_cpu_blocks = max(num_cpu_blocks, 0)
            num_xpu_blocks = 0

            return num_gpu_blocks, num_cpu_blocks, num_xpu_blocks 

        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = total_gpu_memory - free_gpu_memory

        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.model_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        torch.cuda.empty_cache()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, num_cpu_blocks, 0

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        self.cpu_cache = self.cache_engine.cpu_cache
        self.xpu_cache = self.cache_engine.xpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    def warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # Wait for cache operations to finish.
        # TODO(woosuk): Profile swapping overhead and optimize if needed.
        if cache_events is not None:
            for event in cache_events:
                event.wait()
        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            return {}

        kv_caches = None
        if self.model_config.device == torch.device('cpu'):
            kv_caches = self.cpu_cache
        elif self.model_config.device == torch.device('xpu'):
            kv_caches = self.xpu_cache
        else:
            kv_caches = self.gpu_cache

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 kv_caches)
        return output


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        backend = "nccl"
        if parallel_config.device == torch.device('cpu'):
            backend = "gloo"
        if parallel_config.device == torch.device('xpu'):
            backend = "ccl"
            print(f"parallel_config.world_size:{parallel_config.world_size}" )
            print(f"parallel_config.device:{parallel_config.device}" )
            parallel_config.world_size = 1
            # os.environ['RANK'] = str(os.environ.get('RANK', 0))
            # os.environ['WORLD_SIZE'] = str(os.environ.get('WORLD_SIZE', 1))
            os.environ['MASTER_ADDR'] = '127.0.0.1'  # your master address
            os.environ['MASTER_PORT'] = '29500'  # your master port
            # distributed_init_method=os.environ["INIT_FILE"],
            distributed_init_method=None

        torch.distributed.init_process_group(
            backend=backend,
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    # torch.distributed.all_reduce(torch.zeros(1, device=parallel_config.device))
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}.")
