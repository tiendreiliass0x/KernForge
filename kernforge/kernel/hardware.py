"""
GPU hardware specifications used to inform kernel generation.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    name: str
    compute_capability: str
    num_sms: int
    shared_mem_per_sm_kb: int
    l2_cache_mb: int
    hbm_bandwidth_tb_s: float
    fp32_tflops: float
    bf16_tensor_tflops: float
    fp8_tensor_tflops: float
    registers_per_sm: int
    max_threads_per_sm: int
    warp_size: int = 32

    def to_prompt_context(self) -> str:
        return f"""## Target GPU: {self.name}
- Compute Capability: {self.compute_capability}
- SMs: {self.num_sms}
- Shared Memory per SM: {self.shared_mem_per_sm_kb} KB
- L2 Cache: {self.l2_cache_mb} MB
- HBM Bandwidth: {self.hbm_bandwidth_tb_s} TB/s
- FP32 Compute: {self.fp32_tflops} TFLOPS
- BF16 Tensor Core: {self.bf16_tensor_tflops} TFLOPS
- FP8 Tensor Core: {self.fp8_tensor_tflops} TFLOPS
- Registers per SM: {self.registers_per_sm}
- Max Threads per SM: {self.max_threads_per_sm}
- Warp Size: {self.warp_size}

### Key Optimization Principles for {self.name}:
- Shared memory is {self.shared_mem_per_sm_kb}KB — fit your working set here
- Arithmetic intensity threshold: {self.bf16_tensor_tflops * 1000 / self.hbm_bandwidth_tb_s:.0f} ops/byte for BF16 compute-bound
- TMA (Tensor Memory Accelerator): use for async bulk HBM→SMEM copies
- Maximize occupancy: balance register usage vs parallelism
- Prefer bf16 tensor core ops (tl.dot) over scalar f32 when possible
- Use tl.load/tl.store with contiguous access for coalesced memory
"""


# Pre-defined GPU specs
B200 = GPUSpec(
    name="NVIDIA B200",
    compute_capability="10.0",
    num_sms=192,
    shared_mem_per_sm_kb=256,
    l2_cache_mb=64,
    hbm_bandwidth_tb_s=8.0,
    fp32_tflops=90,
    bf16_tensor_tflops=2250,
    fp8_tensor_tflops=4500,
    registers_per_sm=65536,
    max_threads_per_sm=2048,
)

H100 = GPUSpec(
    name="NVIDIA H100",
    compute_capability="9.0",
    num_sms=132,
    shared_mem_per_sm_kb=228,
    l2_cache_mb=50,
    hbm_bandwidth_tb_s=3.35,
    fp32_tflops=67,
    bf16_tensor_tflops=990,
    fp8_tensor_tflops=1979,
    registers_per_sm=65536,
    max_threads_per_sm=2048,
)

GPU_REGISTRY = {
    "B200": B200,
    "H100": H100,
}
