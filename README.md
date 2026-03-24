# 🔥 KernForge

**Agentic GPU kernel generation and evolution system.**

KernForge takes a kernel specification and iteratively generates, evaluates, and optimizes GPU kernels using LLM agents. Built for the [FlashInfer AI Kernel Generation Contest @ MLSys 2026](https://mlsys26.flashinfer.ai/), but designed as a general-purpose kernel optimization pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Evolution Loop                     │
│                                                      │
│  ┌──────────┐   ┌───────────┐   ┌────────────────┐ │
│  │  Agent    │──▶│ Evaluator │──▶│ Analyzer       │ │
│  │ (LLM)    │   │           │   │ (LLM)          │ │
│  │          │◀──│ correct?  │◀──│ bottleneck?    │ │
│  │ generate │   │ fast?     │   │ what to change?│ │
│  └──────────┘   └───────────┘   └────────────────┘ │
│       │              │                  │            │
│       ▼              ▼                  ▼            │
│  ┌──────────────────────────────────────────────┐   │
│  │              Solution History                 │   │
│  │  gen0 → gen1 → gen2 → ... → best             │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
         │                            │
         ▼                            ▼
   Kernel Spec                  Submission
   (FlashInfer-Bench)        (starter-kit format)
```

## Quick Start

```bash
# Install
pip install -e .

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run the contest-compatible smoke config
python -m kernforge.main --config config/config_smoke_test.yaml

# Run the full contest task list locally
python -m kernforge.main --config config/config_deep.yaml --eval_backend local

# Or via the installed entrypoint
kernforge --config config/config_smoke_test.yaml
```

The old `kernforge evolve ...` CLI still exists as `kernforge-legacy`, but the
baseline-compatible runner in `kernforge/main.py` is the canonical path.

## How It Works

### The Loop

1. **Generate**: LLM writes a Triton kernel from the spec + hardware context
2. **Evaluate**: Compile → correctness check → benchmark latency
3. **Fix** (if broken): LLM sees the error and fixes it (up to 3 retries)
4. **Analyze** (every N gens): LLM reviews profiling data, identifies bottleneck
5. **Improve**: LLM generates next version targeting the identified bottleneck
6. **Repeat** until target latency or budget exhausted

### What Makes It Work

- **Structured specs**: The agent gets precise tensor shapes, dtypes, and a reference implementation
- **Hardware context**: The agent knows exact B200 specs (SRAM size, bandwidth, TFLOPS)
- **Error feedback**: Compile errors, runtime errors, and incorrect outputs all get fed back
- **Performance analysis**: Periodically asks the LLM to analyze bottlenecks from profiling data
- **History tracking**: The agent sees what strategies worked/failed in previous generations
- **Single-change discipline**: Each generation makes ONE optimization, making progress legible

## Project Structure

```
kernforge/
├── kernforge/
│   ├── agent/
│   │   ├── hybrid_agent.py  # Hybrid explore/exploit runner
│   │   ├── ladder.py        # Structured optimization ladder
│   │   └── corpus.py        # Reference-kernel corpus
│   ├── eval/
│   │   ├── flashinfer_eval.py # Contest-compatible evaluation path
│   │   ├── modal_eval.py      # Remote B200 evaluation
│   │   └── static_analysis.py # Pre-GPU validation
│   ├── prompt/
│   │   ├── proposer_prompt.py # Initial kernel generation prompt
│   │   └── tuner_prompt.py    # Iterative edit prompt
│   ├── main.py               # Canonical contest runner
│   ├── submit.py             # Submission packer
│   ├── evolve.py             # Legacy spec-driven loop
│   └── cli.py                # Legacy CLI entrypoint
└── pyproject.toml
```

## Extending to Other Tracks

KernForge is track-agnostic. To target a different kernel:

```python
from kernforge.kernel.spec import KernelSpec
from kernforge.evolve import EvolutionLoop

# Load any FlashInfer-Bench definition
spec = KernelSpec.from_flashinfer_bench(
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
    dataset_path="./mlsys26-contest"
)

# Or create a custom spec
spec = KernelSpec.from_file("my_custom_kernel.json")

# Run evolution
loop = EvolutionLoop.create()
state = loop.run(spec)
```

## LLM Backends

| Backend | Config | Best For |
|---------|--------|----------|
| Claude Sonnet 4 | `--backend anthropic --model claude-sonnet-4-20250514` | Default, good balance |
| Claude Opus | `--backend anthropic --model claude-opus-4-20250514` | Complex kernels |
| Local (vLLM/Ollama) | `--backend openai --base-url http://localhost:8000/v1` | No API costs |

## Competition Workflow

```bash
# 1. Clone contest dataset
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest

# 2. Run KernForge on contest definitions
python -m kernforge.main \
    --config config/config_deep.yaml \
    --tasks_path config/tasks_all.txt \
    --eval_backend modal

# 3. Pack outputs into starter-kit / solution.json format
python -m kernforge.submit \
    --output_dir outputs/<run-dir> \
    --name kernforge-v1

# 4. Submit the generated artifacts
```

## Roadmap

- [x] FlashInfer-Bench native integration
- [x] NCU profiling hooks and static analysis
- [x] Parallel candidate generation and strategy learning
- [ ] Unify the legacy `evolve.py` stack with the contest runner
- [ ] Expand true end-to-end smoke coverage across local and Modal backends
- [ ] CUDA backend support (CuTe DSL, CUTLASS)
