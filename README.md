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

# Evolve a GDN decode kernel (Track C)
python examples/evolve_gdn_decode.py --generations 15

# Or use the CLI
kernforge evolve --spec kernforge/templates/gdn_decode_qk16_v32_d128_k_last.json --generations 15

# Use a local model instead
kernforge evolve --spec kernforge/templates/gdn_decode_qk16_v32_d128_k_last.json \
    --backend openai --base-url http://localhost:8000/v1 --model qwen2.5-coder-32b
```

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
│   │   ├── generator.py     # LLM-powered kernel generation (Claude, OpenAI-compat)
│   │   └── prompts.py       # System prompts encoding GPU optimization knowledge
│   ├── eval/
│   │   └── evaluator.py     # Correctness + benchmark evaluation
│   ├── kernel/
│   │   ├── spec.py           # Kernel specification parser
│   │   ├── solution.py       # Solution representation + serialization
│   │   └── hardware.py       # GPU hardware specs (B200, H100)
│   ├── templates/            # Built-in kernel spec templates
│   ├── evolve.py             # Core evolution loop orchestrator
│   └── cli.py                # Command-line interface
├── examples/
│   └── evolve_gdn_decode.py  # Quick-start example
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

# 2. Run evolution
kernforge evolve \
    --definition gdn_decode_qk16_v32_d128_k_last \
    --dataset ./mlsys26-contest \
    --generations 30 \
    --output ./my_submission

# 3. Test on Modal B200
cd my_submission/gdn_decode_qk16_v32_d128_k_last/submission
pip install flashinfer-bench modal
python scripts/run_modal.py

# 4. Submit: share repo URL with organizers
```

## Roadmap

- [ ] FlashInfer-Bench native integration (evaluate via `flashinfer-bench run`)
- [ ] NCU profiling integration for bottleneck analysis
- [ ] Parallel candidate generation (generate N variants, keep best)
- [ ] Multi-track evolution (fused_moe, sparse_attention, gated_delta_net)
- [ ] CUDA backend support (CuTe DSL, CUTLASS)
- [ ] Crossover: combine successful strategies from different lineages
- [ ] Self-play: generate adversarial test cases to find correctness bugs
