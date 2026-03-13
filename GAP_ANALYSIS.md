# KernForge vs Official Baseline: Gap Analysis & Battle Plan

## Baseline Architecture (What We're Beating)

```
agent/main.py          → Task orchestration, resume, result aggregation
agent/iterative_agent.py → Hill-climbing: propose once → str_replace refine N times
agent/evolve_agent.py  → Population: propose N times → softmax-sample elite pool
agent/api.py           → OpenAI/Anthropic with retry + exponential backoff
agent/eval.py          → flashinfer-bench Python API (NOT CLI subprocess)
agent/modal_eval.py    → Same eval on Modal B200 (no local GPU needed)
agent/utils.py         → Task loading, str_replace, code extraction
prompt/proposer_prompt.py → "Write a Triton kernel for {definition}" + kernel pool
prompt/tuner_prompt.py → "Apply str_replace edits" with previous kernels+metrics
```

## What the Baseline Does Right (MUST ADOPT)

### 1. flashinfer-bench Python API (NOT subprocess)
```python
# Baseline uses the API directly — no subprocess, no CLI parsing
from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import TraceSet, Solution, SourceFile, BuildSpec

trace_set = TraceSet.from_path(dataset_root)
solution = Solution(name=..., definition=task_id, spec=BuildSpec(...),
                    sources=[SourceFile(path="main.py", content=kernel_code)])
trace_set.solutions.setdefault(task_id, []).append(solution)
benchmark = Benchmark(trace_set, BenchmarkConfig(warmup_runs=3, iterations=5, ...))
result_ts = benchmark.run_all(dump_traces=False)
```
**WE MUST USE THIS.** Our subprocess-based flashinfer-bench CLI integration is wrong.

### 2. Entry point format
```python
entry_point="main.py::run"  # function "run" in file "main.py"
```
All generated kernels must expose a `run()` function.

### 3. Task definition as JSON (not our KernelSpec)
```python
definition = json.dumps(ref_arch_src, indent=4)  # raw JSON from dataset
# Passed straight into the LLM prompt as {definition}
```
The dataset provides definition JSONs with tensor shapes, dtypes, axes, reference code.

### 4. Modal B200 eval
Their Modal setup is clean — Image from nvidia/cuda, pip install deps, Volume for datasets.
This is the production eval path. No local GPU needed.

### 5. Scoring: (compiled, correct, speedup) tuple
```python
def calculate_score(metric):
    if not metric.compiled: return (0, 0, 0)
    if not metric.correct:  return (1, 0, 0)
    return (1, 1, metric.speedup)
```
Correct > Fast. A correct 0.8x kernel beats an incorrect 2x kernel.

### 6. str_replace editing (iterative agent)
Instead of regenerating the whole kernel each time, the tuner makes targeted edits.
This is more token-efficient and preserves working code.

## What the Baseline Gets Wrong (Our Advantages)

### ❌ 1. ZERO domain knowledge in prompts
Their proposer prompt is essentially:
> "Write a Triton kernel optimized for {gpu_name} for {definition}"

No mention of:
- Common Triton bugs (missing masks, bf16 accumulators, wrong tl.dot dimensions)
- Optimization patterns (tiling, tensor cores, memory coalescing)
- Architecture-specific features (B200 TMA, 256KB SRAM)
- Kernel-type-specific patterns (GDN state update, MOE routing, etc.)

**Our advantage**: 43KB of encoded domain knowledge, optimization ladder, reference corpus.

### ❌ 2. No static analysis (pre-GPU bug catching)
Every kernel goes to flashinfer-bench, even ones with `torch.sigmoid` inside @triton.jit.
Each failed eval costs:
- ~5-10 seconds on local GPU
- ~30-60 seconds on Modal (cold start + eval)
- 1 wasted LLM generation

**Our advantage**: Static analysis catches ~60% of bugs in <0.01 seconds.

### ❌ 3. No bottleneck diagnosis
When a kernel is slow, the baseline only tells the LLM:
> "speedup: 0.7" (or the raw EvalResult)

No information about WHY it's slow. The LLM guesses randomly.

**Our advantage**: NCU profiling → "MEMORY BOUND: 70% DRAM bandwidth, 0% tensor cores, 96 regs/thread"

### ❌ 4. Single candidate per generation
Both agents generate ONE kernel per step. If it's bad, they waste a step.

**Our advantage**: Tournament generates 3-5 diverse candidates, keeps the best.

### ❌ 5. No cross-run learning
Starting fresh every time. No memory of what worked/failed on similar problems.

**Our advantage**: Strategy DB persists across runs. "fuse_output gave 36% speedup on GDN decode."

### ❌ 6. Generic prompts for all kernel types
Same prompt template for GDN, MOE, GEMM, DSA. No kernel-specific guidance.

**Our advantage**: Type-specific prompts, reference implementations, known pitfalls per kernel type.

### ❌ 7. No adversarial test generation
Relies entirely on flashinfer-bench's test suite. If a kernel passes those tests but has
edge-case bugs (non-power-of-2 batch, GVA ratio edge cases), it won't be caught until
the official leaderboard.

**Our advantage**: 11 decode + 15 prefill adversarial test cases per kernel type.

### ❌ 8. Evolve agent has weak diversity
The evolve agent generates from the same prompt with the same temperature.
Diversity comes only from LLM randomness.

**Our advantage**: Tournament with explicit strategy hints ("Focus on MEMORY", "Focus on COMPUTE",
"Try a DIFFERENT ALGORITHM").

### ❌ 9. No optimization ladder
The baseline treats all generations equally. Early generations try random optimizations
instead of following a structured progression.

**Our advantage**: Correctness → Algorithm → Tensor cores → Autotune → Memory → Fusion → Micro-opt

## The Plan: What to Build

### Phase 1: Interface Compatibility (MUST DO FIRST)

**Goal**: KernForge speaks the exact same protocol as the baseline.

1. **Replace our eval with flashinfer-bench Python API**
   - Use `TraceSet.from_path()`, `Solution()`, `Benchmark()` directly
   - Drop our custom `_test_correctness` for the real evaluator
   - Match their `EvalResult(compiled, correct, speedup, latency_ms, error, stats)`

2. **Replace our KernelSpec with their task loading**
   - `load_flashinfer_trace_definition(op_type, problem_id)` → raw JSON
   - Pass definition JSON directly in prompts
   - Entry point: `main.py::run`

3. **Modal B200 eval**
   - Adopt their Modal app pattern
   - Ensure our Modal function matches their Volume + Image setup

4. **Config compatibility**
   - Support their YAML config format
   - Support their task list format (`dsa_paged\ngdn\nmoe`)

5. **Output format**
   - `outputs/{agent}_{source}_{steps}_{timestamp}/{level}_{problem_id}/`
   - `global_best_kernel_N.py` + `global_best_metrics_N.json`

### Phase 2: Inject Our Advantages Into Their Framework

**Goal**: Same interface, dramatically better kernel quality.

6. **Static analysis gate (before flashinfer-bench)**
   ```
   generate kernel → static_analysis.analyze() → if blocking: skip eval, feed back
   ```
   Saves 60% of wasted eval round-trips.

7. **Domain knowledge in prompts**
   - Inject our optimization ladder context based on generation number
   - Add kernel-type-specific guidance (GDN, MOE, GEMM, DSA)
   - Include common bug patterns as negative examples

8. **Reference corpus in prompts**
   - For GDN: inject our annotated decode/prefill references
   - For others: fetch from FLA/FlashInfer repos
   - Format as few-shot examples in the proposer prompt

9. **NCU bottleneck feedback in tuner prompt**
   - After eval, if kernel is correct but slow, run NCU
   - Include bottleneck diagnosis in the next tuner prompt
   - "Your kernel is MEMORY BOUND at 70% bandwidth. Consider..."

10. **Tournament selection**
    - Replace single `propose_step` with 3-5 parallel proposals
    - Each with different strategy hints
    - Keep best by `calculate_score()`

11. **Strategy DB integration**
    - Load proven strategies before first proposal
    - Record outcomes after each generation
    - Include in prompt: "✅ fuse_output: 36% faster" / "❌ wrong_mask: runtime_error"

### Phase 3: Advanced Optimizations

12. **Hybrid agent**: Start with evolve (explore), switch to iterative (exploit) once
    a correct kernel exists. Best of both worlds.

13. **Parallel eval**: When tournament generates 5 candidates, evaluate all 5 in parallel
    on Modal (5 separate function calls).

14. **Adaptive step allocation**: If a problem already has speedup > 1.5x after 5 steps,
    move on to harder problems. Spend more steps on problems where we're stuck.

15. **Problem-type routing**: Use different agent configs per problem type:
    - GDN: iterative with heavy domain knowledge, ladder from correctness
    - GEMM: evolve with autotune exploration, many configs
    - MOE: hybrid, focus on routing + fusion
    - DSA: iterative, focus on paged attention patterns

## Concrete File Plan

```
kernforge/
  agent/
    main.py              # Compatible with baseline CLI + our enhancements
    iterative_agent.py   # Their loop + static analysis + NCU + domain knowledge
    evolve_agent.py      # Their loop + tournament + strategy DB
    hybrid_agent.py      # NEW: evolve → iterative transition
    api.py               # Their API client (already works)
    eval.py              # flashinfer-bench Python API (replace our subprocess)
    modal_eval.py        # Their Modal setup (adopt directly)
  prompt/
    proposer_prompt.py   # Their base + our domain knowledge injection
    tuner_prompt.py      # Their base + NCU feedback + ladder context
    domain_knowledge.py  # Our GPU kernel engineering knowledge
  eval/
    static_analysis.py   # Keep ours (pre-GPU filtering)
    ncu.py              # Keep ours (bottleneck diagnosis)
    adversarial.py      # Keep ours (edge case testing)
    tournament.py       # Keep ours (multi-candidate selection)
  agent/
    corpus.py           # Keep ours (reference implementations)
    ladder.py           # Keep ours (optimization phases)
    strategy_db.py      # Keep ours (cross-run learning, rename from tournament.py)
  config/
    tasks_default.txt   # Same as baseline
    config_kernforge.yaml # Our enhanced config
```

## Expected Impact

| Component | Baseline | KernForge | Expected Improvement |
|-----------|----------|-----------|---------------------|
| Bugs caught before GPU | 0% | ~60% | 2-3x fewer wasted evals |
| Candidates per generation | 1 | 3-5 | 2-3x better per-gen winner |
| Domain knowledge | None | 43KB structured | Higher first-attempt quality |
| Bottleneck diagnosis | "speedup: 0.7" | "MEMORY BOUND, 70% BW" | Targeted optimizations |
| Cross-run learning | None | Strategy DB | Better starting point each run |
| Kernel-specific guidance | None | Per-type prompts + refs | Fewer correctness bugs |
| Steps to correct kernel | ~5-10 | ~2-3 | 2-3x faster convergence |
| Steps to target speedup | ~15-25 | ~8-15 | 40-50% fewer steps |

## Priority Order

1. **flashinfer-bench Python API integration** (blocks everything else)
2. **Task loading compatibility** (blocks testing)
3. **Static analysis gate** (highest ROI: free, catches 60% of bugs)
4. **Domain knowledge in prompts** (second highest ROI: no extra API calls)
5. **Tournament selection** (3x more candidates per eval budget)
6. **NCU bottleneck feedback** (turns "slow" into actionable guidance)
7. **Strategy DB** (compounds over runs)
8. **Modal B200 eval** (needed for production)
9. **Hybrid agent** (explore → exploit transition)
10. **Parallel eval on Modal** (throughput optimization)
