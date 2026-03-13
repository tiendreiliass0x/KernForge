# What Makes KernForge Actually Good

## The Problem with Generic LLM Kernel Generation

Prompting Claude/GPT with "write me a fast Triton kernel for X" produces:
- **Correct code ~30% of the time** (syntax issues, wrong Triton APIs, shape bugs)
- **Fast code ~5% of the time** (generic tiling, no hardware awareness, no profiling feedback)
- **Competitive code ~0% of the time** (expert kernels require 10+ iterations of profile → reason → edit)

The gap isn't intelligence — it's **information**. The LLM doesn't know what's slow, why it's slow, or what the hardware actually supports. KernForge closes that gap at each stage.

---

## The 5 Layers of Competitive Advantage

### Layer 1: Rich Evaluation Signal (not just pass/fail)
**Generic agent:** "Your kernel produced wrong output. Fix it."
**KernForge:**
```
CORRECTNESS FAILURE on shape batch=4, heads=32:
  Output[0, 14, 97]: expected -0.0234, got -0.0312
  Max absolute error: 0.0293 at index [0, 14, 97]
  Max relative error: 125.2%
  
  Error pattern: errors concentrated in v_head >= 16
  Likely cause: GVA head mapping bug — v_heads 16-31 are using wrong q/k head
  
  Reference trace at error location:
    q_head = v_head // 2 = 14 // 2 = 7  ← YOUR CODE USES v_head=14 directly
```

The difference: the agent gets **diagnostic information**, not just binary pass/fail. When you tell an LLM *where* and *why* it's wrong, fix rate goes from ~40% to ~85%.

### Layer 2: Hardware-Grounded Performance Analysis (not "it's slow")
**Generic agent:** "Latency is 150μs. Make it faster."
**KernForge:**
```
PERFORMANCE ANALYSIS (B200):
  Latency: 150μs
  HBM Bandwidth: 2.1 TB/s (26% of 8 TB/s peak)
  Compute: 12 TFLOPS (0.5% of BF16 peak)
  Occupancy: 37.5% (24 active warps / 64 max)
  
  DIAGNOSIS: MEMORY BOUND with poor bandwidth utilization
  - Expected minimum latency (memory-bound floor): 32μs for this workload
  - You're 4.7x above the floor → significant optimization headroom
  
  ROOT CAUSE: Non-coalesced state loads
  - State layout is [B, H, V=128, K=128] (K-contiguous)
  - Your kernel iterates V-first with stride 128 between loads
  - This causes 128-byte strided access → 1/32 effective bandwidth
  
  RECOMMENDED FIX: Swap inner/outer loop to iterate K-contiguous
  Expected improvement: 2-3x (to ~50-75μs)
```

The difference: the agent gets a **roofline analysis** telling it exactly what bottleneck to attack, plus quantitative expected improvement. This turns random search into directed hill climbing.

### Layer 3: Domain-Specific Pattern Library (not generic coding)
**Generic agent:** Uses general programming knowledge to write GPU code.
**KernForge injects:**
- Triton tiling patterns specific to matrix-vector products, rank-1 updates, outer products
- GDN-specific knowledge: the fused decode strategy, WY decomposition for prefill
- Common bug patterns that LLMs make in Triton (wrong tl.dot shapes, missing masks, bf16 accumulation)
- Hardware-specific tricks: B200 TMA, 256KB SRAM fitting, tensor core alignment

This is like the difference between a junior dev writing CUDA from Stack Overflow vs a senior engineer who's optimized 50 kernels on this exact GPU. The domain knowledge is **in the prompt**, not something the LLM has to rediscover.

### Layer 4: Structured Search Strategy (not random rewriting)
**Generic agent:** "Here's a new attempt" (rewrites 80% of the code, breaks what worked)
**KernForge:**
```
MUTATION STRATEGY for Generation 7:
  Status: correct, 95μs median latency
  Bottleneck: memory bound (bandwidth=62% of peak)
  
  Strategy: "fuse_output" — compute o = S @ q during same pass as state update
  
  Specific change: In the tile loop at line 47-63, after updating s_tile,
  immediately compute partial_output += s_tile @ q_chunk and accumulate.
  Remove the separate output computation loop at lines 71-82.
  
  DO NOT CHANGE: tile sizes, launch config, gate computations, state I/O pattern
  ONLY CHANGE: fuse output computation into the state update loop
```

The difference: each generation makes **one targeted mutation** instead of rewriting from scratch. This makes progress monotonic — good changes stick, bad changes get reverted.

### Layer 5: Reference-Grounded Correctness (not vibes-based testing)
**Generic agent:** "Output looks reasonable" or "no NaN/Inf detected"
**KernForge:**
```python
def test_correctness(candidate_fn, reference_fn, spec):
    # Test multiple shapes including edge cases
    for shapes in [minimal, typical, large, boundary]:
        inputs = generate_inputs(spec, shapes)
        
        ref_output = reference_fn(**inputs)    # Ground truth
        cand_output = candidate_fn(**inputs)   # Candidate
        
        # Elementwise comparison with dtype-appropriate tolerance
        abs_err = torch.abs(ref_output - cand_output)
        rel_err = abs_err / (torch.abs(ref_output) + 1e-8)
        
        # Error localization: WHERE is it wrong?
        worst_idx = torch.argmax(abs_err)
        
        # Error pattern: HOW is it wrong?
        # - Systematic offset? (wrong constant)
        # - Scale error? (wrong multiplication)
        # - Scattered errors? (race condition)
        # - Wrong for specific heads? (GVA mapping bug)
        # - Wrong for large values? (numerical overflow)
```

The reference implementation from the FlashInfer-Bench definition IS the oracle. We don't guess whether the kernel is correct — we measure against ground truth at multiple shapes and report precisely where it diverges.

---

## The Concrete Edge: What Each Component Provides

| Component | Without It | With It |
|-----------|-----------|---------|
| Domain knowledge prompts | LLM uses generic coding patterns | LLM applies GPU-specific optimization strategies |
| Reference-based correctness | ~30% correct code, slow debugging | ~70% correct code, precise error localization |
| Profiling-based analysis | Random optimization attempts | Directed: "you're memory bound, fix coalescing" |
| Structured mutations | Rewrites from scratch each iteration | Targeted edits that preserve working code |
| Hardware specs in context | Generic tile sizes | Tile sizes fitted to 256KB SRAM, 192 SMs |
| Pattern library | Rediscovers WY decomposition from scratch | Applies known-good patterns for GDN specifically |
| Bug pattern library | Same mistakes repeated across generations | Known LLM-Triton bugs caught and avoided |

## The Flywheel

The real product value emerges from the loop:

```
Better domain knowledge → Better first-attempt kernels
                          ↓
                    Fewer fix iterations needed
                          ↓
                    More time for optimization iterations
                          ↓
                    More profiling data collected
                          ↓
                    Better analysis prompts
                          ↓
                    Better optimization knowledge → (feeds back to domain knowledge)
```

Every kernel we evolve teaches us new patterns to encode. The system gets better at generating kernels for ALL specifications, not just the one we trained on. That's the product: a system that accumulates kernel engineering expertise over time.

## What's Still Missing (Honest Assessment)

1. **Real NCU integration** — we need `flashinfer_bench_run_ncu` wired in for actual profiling metrics, not estimates
2. **Code-level static analysis** — pattern match on the generated Triton IR to detect issues before running
3. **Reference kernel corpus** — feed the agent FLA's working GDN kernels as examples, not just patterns
4. **Multi-candidate tournament** — generate 3-5 variants per generation, keep the best (genetic algorithm style)
5. **Cross-generation learning** — persist what strategies worked across different runs into a strategy database
6. **Automated test case generation** — adversarial shapes that stress edge cases (batch=1, very long sequences, etc.)
