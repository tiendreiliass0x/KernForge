"""
Reference Kernel Corpus — working code examples the agent can learn from.

Instead of the agent writing GDN kernels from first principles, give it
REAL working Triton implementations as few-shot examples. These come from:

1. FLA (Flash Linear Attention) — fla-org/flash-linear-attention
   - Production GDN (Gated Delta Rule) kernels
   - Triton implementations of chunk/fused attention variants

2. FlashInfer — flashinfer-ai/flashinfer
   - Reference implementations in the contest dataset

3. Community kernels — high-quality open-source implementations

The corpus is loaded lazily (fetched from GitHub on first use) and cached
locally. Each reference includes the source + structured annotations
explaining what it does and why specific design choices were made.
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Default cache directory
CORPUS_DIR = Path.home() / ".kernforge" / "corpus"


@dataclass
class ReferenceKernel:
    """A reference kernel with annotations."""
    name: str
    source: str
    description: str
    kernel_type: str  # "gdn_decode", "gdn_prefill", "attention", etc.
    annotations: str  # structured explanation of design choices
    origin: str  # "fla", "flashinfer", "community"
    url: str = ""

    def to_few_shot_prompt(self, max_lines: int = 150) -> str:
        """Format as a few-shot example for the LLM."""
        # Trim source if too long
        lines = self.source.split("\n")
        if len(lines) > max_lines:
            source = "\n".join(lines[:max_lines]) + f"\n# ... ({len(lines) - max_lines} more lines)"
        else:
            source = self.source

        return f"""## Reference Implementation: {self.name}
Origin: {self.origin} ({self.url})
{self.description}

### Key Design Choices
{self.annotations}

### Source Code
```python
{source}
```
"""


# =============================================================================
# Built-in reference snippets (always available, no network needed)
# =============================================================================

# These are simplified but correct patterns extracted from FLA's implementations.
# They show the KEY design decisions without the full production complexity.

GDN_DECODE_REFERENCE = ReferenceKernel(
    name="GDN Decode (Fused State Update)",
    source='''import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BV': 32}, num_warps=4, num_stages=2),
        triton.Config({'BV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BV': 64}, num_warps=8, num_stages=2),
        triton.Config({'BV': 128}, num_warps=8, num_stages=1),
    ],
    key=['V', 'K'],
)
@triton.jit
def gdn_decode_kernel(
    # Pointers
    q_ptr, k_ptr, v_ptr, state_ptr, new_state_ptr, output_ptr,
    alpha_ptr, beta_ptr, dt_ptr, scale_ptr,
    # Strides
    stride_sb, stride_sh, stride_sv, stride_sk,
    # Dimensions
    B, H_q, H_v, K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr,
):
    """Fused GDN decode: state update + output in one kernel."""
    # Grid: (batch, v_head, num_v_tiles)
    i_b = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_tv = tl.program_id(2)

    # GVA mapping: v_head -> q/k head
    gva_ratio = V // K  # typically 2 for GDN
    i_hq = i_hv // (H_v // H_q)

    # Load inputs (small, fits in registers)
    offs_k = tl.arange(0, K)
    q = tl.load(q_ptr + i_b * H_q * K + i_hq * K + offs_k).to(tl.float32)
    k = tl.load(k_ptr + i_b * H_q * K + i_hq * K + offs_k).to(tl.float32)

    offs_v = i_tv * BV + tl.arange(0, BV)
    v_mask = offs_v < V
    v = tl.load(v_ptr + i_b * H_v * V + i_hv * V + offs_v, mask=v_mask, other=0.0).to(tl.float32)

    # Load gate values
    alpha = tl.load(alpha_ptr + i_hq).to(tl.float32)  # per-head decay
    beta = tl.load(beta_ptr + i_b * H_q + i_hq).to(tl.float32)  # per-token gate
    dt = tl.load(dt_ptr + i_b * H_v + i_hv).to(tl.float32)  # per-head delta step
    scale = tl.load(scale_ptr).to(tl.float32)

    # Precompute: reduces per-element ops in tile loop
    ak = alpha * beta * k  # for householder subtract
    dk = dt * k            # for delta write

    # Load state tile: [BV, K], f32
    state_offset = i_b * stride_sb + i_hv * stride_sh
    s_ptrs = state_ptr + state_offset + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
    s_mask = v_mask[:, None]

    state_tile = tl.load(s_ptrs, mask=s_mask, other=0.0)

    # Fused update:
    # 1. S @ k → partial dot product (reduction over K)
    sk = tl.sum(state_tile * k[None, :], axis=1)  # [BV]

    # 2. Decay + Householder erase + Delta write
    state_tile = alpha * state_tile  # decay
    state_tile = state_tile - sk[:, None] * ak[None, :]  # erase
    state_tile = state_tile + v[:, None] * dk[None, :]  # write

    # 3. Output: S_new @ q
    output_partial = tl.sum(state_tile * q[None, :], axis=1) * scale  # [BV]

    # Store updated state (f32)
    tl.store(s_ptrs, state_tile, mask=s_mask)

    # Atomically accumulate output (different v_tiles contribute to same output)
    # OR: if BV == V, just store directly
    out_ptrs = output_ptr + i_b * H_v * V + i_hv * V + offs_v
    tl.store(out_ptrs, output_partial.to(tl.bfloat16), mask=v_mask)


def gdn_decode(q, k, v, state, alpha, beta, dt, scale):
    """Python wrapper for GDN decode kernel."""
    B, H_q, K = q.shape
    _, H_v, V = v.shape

    output = torch.empty(B, H_v, V, dtype=torch.bfloat16, device=q.device)
    new_state = state.clone()

    grid = lambda meta: (B, H_v, triton.cdiv(V, meta['BV']))
    gdn_decode_kernel[grid](
        q, k, v, new_state, new_state, output,
        alpha, beta, dt, scale,
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        B, H_q, H_v, K, V,
    )
    return output, new_state
''',
    description="Fused GDN decode with tiled state update. Key: all state operations (decay, erase, write, output) in one pass per tile.",
    kernel_type="gdn_decode",
    annotations="""
1. **Grid design**: (batch, v_head, v_tile) — one block per state tile
2. **GVA mapping**: i_hq = i_hv // (H_v // H_q) — integer division, no repeat_interleave
3. **Precompute gates**: ak = alpha * beta * k and dk = dt * k computed once, reused per tile element
4. **Fused operations**: decay + householder + delta write in single pass over state tile
5. **Mixed precision**: inputs loaded as bf16, cast to f32 for state arithmetic, output cast back to bf16
6. **State tiling**: BLOCK_V tiles over rows (V dimension), full K=128 per tile
7. **Autotune**: 4 configs varying BV and num_warps, keyed on V and K
8. **Memory pattern**: K-last layout means K dimension is contiguous → coalesced loads
""",
    origin="fla (adapted)",
    url="https://github.com/fla-org/flash-linear-attention",
)


GDN_CHUNK_REFERENCE = ReferenceKernel(
    name="GDN Chunk Attention (Intra-chunk)",
    source='''# Key insight: within a chunk of C tokens, the delta rule attention
# can be decomposed into:
#   1. A causal attention-like term (parallel over chunk)
#   2. A cross-chunk term from the inter-chunk state
#
# The causal term uses WY representation to express
# the product of Householder-like matrices:
#   P = prod_t (I - beta_t * k_t * k_t^T) = I - W @ Y^T
#
# Where W, Y in R^{d x C} are built incrementally:
#   Y[:, t] = k_t
#   W[:, t] = beta_t * (I - W[:,:t] @ Y[:,:t]^T) @ k_t

# The intra-chunk output for token t is:
#   o_t = sum_{s<=t} [gamma_{t,s} * (alpha_s * beta_s * <k_s, S_{s-1} @ q_t> 
#         - dt_s * <k_s, v_s> * <k_s, q_t>) + dt_s * <v_s, q_t>]
#
# where gamma_{t,s} = prod_{r=s+1}^{t} alpha_r * (1 - beta_r * <k_r, k_r>)

# This is computed as a causal matrix multiplication:
#   A[t, s] = <q_t, k_s>  (attention scores)
#   Masked + scaled by gamma factors
#   Output = A @ V (chunk-local "values" including state contribution)
''',
    description="Chunkwise parallel GDN: WY decomposition + intra-chunk attention. This is the PREFILL algorithm.",
    kernel_type="gdn_prefill",
    annotations="""
1. **WY decomposition**: Avoids sequential d×d matrix products — instead builds W,Y incrementally (O(C) rank-1 updates)
2. **Chunk size C**: Typically 64 — small enough for intra-chunk attention (64²=4K), large enough to amortize inter-chunk cost
3. **Three-phase computation**: (1) build WY factors, (2) intra-chunk attention, (3) inter-chunk state scan
4. **Parallelism**: Intra-chunk attention is fully parallel (like standard attention within chunk)
5. **The inter-chunk scan is sequential** but only O(L/C) steps instead of O(L)
""",
    origin="fla (conceptual)",
    url="https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule",
)


SOFTMAX_ATTENTION_REFERENCE = ReferenceKernel(
    name="Flash Attention Decode (for comparison)",
    source='''# Standard flash attention decode pattern (for comparison with GDN)
# Key differences from GDN:
#   - No persistent state (stateless)
#   - Softmax normalization (GDN uses linear attention)
#   - No decay/gate mechanism
#
# But the TILING STRATEGY is similar:
#   - Tile over K/V sequence length
#   - Keep Q in registers (single token for decode)
#   - Accumulate output + running max/sum for softmax

@triton.jit
def flash_decode_kernel(Q, K, V, Out, stride_kn, stride_kd, N, D: tl.constexpr, BD: tl.constexpr):
    # Single query token, tile over KV cache length
    offs_d = tl.arange(0, BD)
    q = tl.load(Q + offs_d, mask=offs_d < D)

    m_prev = float('-inf')
    l_prev = 0.0
    acc = tl.zeros([BD], dtype=tl.float32)

    for start in range(0, N, BD):
        offs_n = start + tl.arange(0, BD)
        mask_n = offs_n < N

        k = tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=mask_n[:, None])
        v = tl.load(V + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=mask_n[:, None])

        # Attention scores
        s = tl.sum(q[None, :] * k, axis=1)  # [BD]

        # Online softmax
        m_new = tl.maximum(m_prev, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        l_new = l_prev * tl.exp(m_prev - m_new) + tl.sum(p)

        # Rescale and accumulate
        acc = acc * (l_prev * tl.exp(m_prev - m_new) / l_new)
        acc += tl.sum(p[:, None] * v, axis=0) / l_new

        m_prev = m_new
        l_prev = l_new

    tl.store(Out + offs_d, acc.to(tl.bfloat16), mask=offs_d < D)
''',
    description="Flash attention decode for comparison. Shows tiling-over-sequence pattern that GDN adapts for state-based attention.",
    kernel_type="attention",
    annotations="""
Useful comparison point: GDN decode tiles over STATE dimensions, not sequence length.
The online accumulation pattern (running max/sum) is analogous to GDN's sequential state update.
""",
    origin="community",
    url="",
)


MOE_REFERENCE = ReferenceKernel(
    name="Fused MOE (Expert Routing + Grouped GEMM)",
    source='''import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['N', 'K'],
)
@triton.jit
def fused_moe_kernel(
    # Input pointers
    x_ptr, w_ptr, out_ptr,
    # Routing pointers
    topk_ids_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_pad_ptr,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    # Dimensions
    N: tl.constexpr, K: tl.constexpr, num_experts: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    top_k: tl.constexpr,
):
    """Fused MOE: route tokens to experts + grouped GEMM in one kernel.

    Grid: (num_token_blocks, num_N_blocks, num_experts)
    Each block computes a BLOCK_M x BLOCK_N tile of the output for one expert.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)

    # Load number of tokens assigned to this expert
    num_tokens = tl.load(num_tokens_post_pad_ptr + pid_e)
    if pid_m * BLOCK_M >= num_tokens:
        return  # early exit — no work for this block

    # Load sorted token indices for this expert
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Token IDs assigned to this expert (sorted for coalescing)
    token_ids = tl.load(sorted_token_ids_ptr + pid_e * BLOCK_M + offs_m, mask=offs_m < num_tokens, other=0)

    # Accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Tiled GEMM: x[tokens, :] @ w[expert, :, :].T
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # Load input tile (gather by token ID)
        x_tile = tl.load(
            x_ptr + token_ids[:, None] * stride_xm + k_offs[None, :] * stride_xk,
            mask=(offs_m[:, None] < num_tokens) & (k_offs[None, :] < K),
            other=0.0,
        )

        # Load weight tile for this expert
        w_tile = tl.load(
            w_ptr + pid_e * N * K + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (k_offs[None, :] < K),
            other=0.0,
        )

        # Tensor core GEMM
        acc += tl.dot(x_tile, tl.trans(w_tile))

    # Apply routing weight and scatter to output
    routing_weights = tl.load(topk_weights_ptr + token_ids, mask=offs_m < num_tokens, other=0.0)
    acc = acc * routing_weights[:, None]

    # Store output (scatter by token ID)
    tl.store(
        out_ptr + token_ids[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < N),
    )
''',
    description="Fused MOE kernel: expert routing + grouped GEMM. Key: tokens sorted by expert for coalesced access, one kernel does both routing and compute.",
    kernel_type="moe",
    annotations="""
1. **Grid design**: (token_blocks, N_blocks, experts) — each expert gets its own z-dimension
2. **Token sorting**: tokens pre-sorted by expert assignment for coalesced memory access
3. **Early exit**: blocks with no tokens skip immediately (critical for load balance)
4. **Gather-compute-scatter**: gather input by token ID, GEMM with expert weights, scatter output
5. **Routing weight fusion**: multiply by topk_weight inside the kernel (avoids separate pass)
6. **Tensor cores**: tl.dot for the main GEMM — compute-bound kernel
7. **Mixed precision**: f32 accumulation, bf16 inputs and outputs
8. **Autotune**: varies BLOCK_M/N/K and warps, keyed on N and K dimensions
""",
    origin="community (vLLM-style)",
    url="https://github.com/vllm-project/vllm",
)


SPARSE_ATTENTION_REFERENCE = ReferenceKernel(
    name="Paged Attention Decode (Page Table Lookup + Tiled KV Cache)",
    source='''import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_KV': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_KV': 128, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_KV': 128, 'BLOCK_D': 128}, num_warps=8, num_stages=2),
    ],
    key=['D', 'max_seq_len'],
)
@triton.jit
def paged_attention_decode_kernel(
    # Pointers
    q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
    page_table_ptr, seq_lens_ptr,
    # Strides
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kd,  # cache: [num_blocks, block_size, D]
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_ptb, stride_ptn,  # page table: [batch, max_pages]
    # Dimensions
    num_heads: tl.constexpr, D: tl.constexpr,
    page_size: tl.constexpr, max_seq_len: tl.constexpr,
    BLOCK_KV: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Paged attention decode: single query token attends to paged KV cache.

    The KV cache is stored in fixed-size pages (e.g., 16 tokens each).
    A page table maps (batch, logical_page_idx) -> physical_page_id.
    This avoids memory fragmentation from variable-length sequences.
    """
    i_b = tl.program_id(0)
    i_h = tl.program_id(1)

    # Load this sequence's length
    seq_len = tl.load(seq_lens_ptr + i_b)

    # Load query (single token, D dimensions)
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr + i_b * stride_qb + i_h * stride_qh + offs_d * stride_qd,
        mask=offs_d < D, other=0.0,
    ).to(tl.float32)

    # Online softmax state
    m_prev = float('-inf')
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Iterate over KV cache pages
    num_pages = tl.cdiv(seq_len, page_size)
    for page_idx in range(0, tl.cdiv(max_seq_len, page_size)):
        if page_idx >= num_pages:
            break

        # Page table lookup: logical page -> physical block
        phys_page = tl.load(page_table_ptr + i_b * stride_ptb + page_idx * stride_ptn)

        # Load K and V from this page
        base_kv_idx = page_idx * page_size
        offs_kv = tl.arange(0, BLOCK_KV)
        kv_mask = (base_kv_idx + offs_kv) < seq_len

        # K: [page_size, D]
        k = tl.load(
            k_cache_ptr + phys_page * stride_kb + offs_kv[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=kv_mask[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        # Attention scores: q @ k.T
        s = tl.sum(q[None, :] * k, axis=1)  # [BLOCK_KV]
        s = tl.where(kv_mask, s, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_prev, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        l_new = l_prev * tl.exp(m_prev - m_new) + tl.sum(p)

        # V: [page_size, D]
        v = tl.load(
            v_cache_ptr + phys_page * stride_vb + offs_kv[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=kv_mask[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        # Rescale and accumulate
        acc = acc * (l_prev * tl.exp(m_prev - m_new) / tl.maximum(l_new, 1e-6))
        acc += tl.sum(p[:, None] * v, axis=0) / tl.maximum(l_new, 1e-6)

        m_prev = m_new
        l_prev = l_new

    # Store output
    tl.store(
        out_ptr + i_b * stride_ob + i_h * stride_oh + offs_d * stride_od,
        acc.to(tl.bfloat16),
        mask=offs_d < D,
    )
''',
    description="Paged attention decode: page table lookup + tiled KV cache access with online softmax. Key pattern for memory-efficient inference.",
    kernel_type="dsa_paged",
    annotations="""
1. **Page table indirection**: logical page index -> physical block via page_table lookup
2. **Variable-length handling**: seq_lens per batch, early exit when page_idx >= num_pages
3. **Online softmax**: running max + sum for numerical stability across pages
4. **Grid design**: (batch, heads) — each block handles one query-head pair
5. **Memory layout**: KV cache stored in fixed-size pages, avoids fragmentation
6. **Masking**: kv_mask handles partial last page (seq_len not divisible by page_size)
7. **Mixed precision**: f32 for attention computation, bf16 for I/O
8. **Autotuned**: BLOCK_KV and BLOCK_D varied, keyed on D and max_seq_len
""",
    origin="community (vLLM-style paged attention)",
    url="https://github.com/vllm-project/vllm",
)


# =============================================================================
# Corpus management
# =============================================================================

# All built-in references
BUILTIN_CORPUS = {
    "gdn_decode": [GDN_DECODE_REFERENCE],
    "gdn_prefill": [GDN_CHUNK_REFERENCE],
    "attention": [SOFTMAX_ATTENTION_REFERENCE],
    "moe": [MOE_REFERENCE],
    "dsa_paged": [SPARSE_ATTENTION_REFERENCE],
}


def get_references(kernel_type: str, max_refs: int = 2) -> list[ReferenceKernel]:
    """Get relevant reference kernels for a kernel type."""
    refs = []

    # Exact match
    if kernel_type in BUILTIN_CORPUS:
        refs.extend(BUILTIN_CORPUS[kernel_type])

    # Related matches
    if "gdn" in kernel_type or "gated_delta" in kernel_type:
        for key in ("gdn_decode", "gdn_prefill"):
            if key != kernel_type and key in BUILTIN_CORPUS:
                refs.extend(BUILTIN_CORPUS[key])

    if "moe" in kernel_type or "expert" in kernel_type:
        if "moe" in BUILTIN_CORPUS and kernel_type != "moe":
            refs.extend(BUILTIN_CORPUS["moe"])

    if "dsa" in kernel_type or "paged" in kernel_type or "sparse_attention" in kernel_type:
        if "dsa_paged" in BUILTIN_CORPUS and kernel_type != "dsa_paged":
            refs.extend(BUILTIN_CORPUS["dsa_paged"])

    # Always include attention for comparison
    if kernel_type != "attention" and "attention" in BUILTIN_CORPUS:
        refs.extend(BUILTIN_CORPUS["attention"])

    # Check for cached external references
    cached = _load_cached_references(kernel_type)
    refs.extend(cached)

    return refs[:max_refs]


def references_to_prompt(refs: list[ReferenceKernel]) -> str:
    """Format references as few-shot context for the LLM."""
    if not refs:
        return ""

    sections = ["## Reference Implementations\n"
                 "Study these working kernels before writing your own. "
                 "Pay attention to the design choices annotated below.\n"]

    for ref in refs:
        sections.append(ref.to_few_shot_prompt())

    return "\n".join(sections)


def fetch_fla_references(cache_dir: Path | None = None) -> list[ReferenceKernel]:
    """
    Fetch FLA's GDN kernel implementations from GitHub.
    Caches locally for offline use.
    """
    cache_dir = cache_dir or CORPUS_DIR / "fla"
    cache_dir.mkdir(parents=True, exist_ok=True)

    refs = []
    fla_files = [
        ("fla/ops/gated_delta_rule/fused_recurrent.py", "gdn_decode"),
        ("fla/ops/gated_delta_rule/chunk.py", "gdn_prefill"),
        ("fla/ops/gated_delta_rule/wy_fast.py", "gdn_prefill"),
        ("fla/ops/delta_rule/fused_recurrent.py", "gdn_decode"),
        ("fla/ops/gla/chunk.py", "gdn_prefill"),
    ]

    for filepath, ktype in fla_files:
        cached = cache_dir / filepath.replace("/", "_")
        if cached.exists():
            source = cached.read_text()
        else:
            try:
                url = f"https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/{filepath}"
                result = subprocess.run(
                    ["curl", "-sL", url], capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0 and len(result.stdout) > 100:
                    source = result.stdout
                    cached.write_text(source)
                else:
                    continue
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        refs.append(ReferenceKernel(
            name=f"FLA {filepath.split('/')[-1]}",
            source=source,
            description=f"Production FLA implementation from {filepath}",
            kernel_type=ktype,
            annotations="(Auto-fetched — see source for inline comments)",
            origin="fla",
            url=f"https://github.com/fla-org/flash-linear-attention/blob/main/{filepath}",
        ))

    return refs


def _load_cached_references(kernel_type: str) -> list[ReferenceKernel]:
    """Load any cached external references."""
    cache_dir = CORPUS_DIR / kernel_type
    if not cache_dir.exists():
        return []

    refs = []
    for f in cache_dir.glob("*.py"):
        refs.append(ReferenceKernel(
            name=f.stem,
            source=f.read_text(),
            description=f"Cached reference from {f}",
            kernel_type=kernel_type,
            annotations="",
            origin="cached",
        ))
    return refs
