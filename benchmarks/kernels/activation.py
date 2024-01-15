import torch
from benchmark import KernelBenchmark

from vllm._C import ops

class ActivationBench(KernelBenchmark):

    def __init__(self, loop_time, num_tokens, d, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__(loop_time)
        self.num_tokens = num_tokens
        self.d = d
        self.input = torch.randn(num_tokens, 2 * d, dtype=dtype, device=device)
        self.output = torch.empty(num_tokens, d, dtype=dtype, device=device)

    def _run(self):
        for i in range(self.loop_time):
            ops.silu_and_mul(self.output, self.input)


bench = ActivationBench(10, 4096, 512, torch.float32, torch.device("xpu"))
bench.execute()

# RMSNormBench(10, 4096, 4096, torch.float32, torch.device("cpu"))
# Scalar: 282420151.5 ns
# token parallel: 36635991.875 ns 7.7x
# FMA: 36517116.125 ns