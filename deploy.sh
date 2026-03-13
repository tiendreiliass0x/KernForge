#!/usr/bin/env bash
# KernForge Competition Deploy Script
# One-command setup and run for MLSys26 FlashInfer AI Kernel Generation Contest
#
# Usage:
#   # Full setup (first time)
#   ./deploy.sh setup
#
#   # Run on Modal B200 (no local GPU needed)
#   ./deploy.sh run-modal --track gated_delta_net
#
#   # Run on local GPU
#   ./deploy.sh run-local --track fused_moe
#
#   # Run all tracks
#   ./deploy.sh run-all-modal
#
#   # Pack submission
#   ./deploy.sh submit --track gated_delta_net

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${SCRIPT_DIR}/datasets/mlsys26-contest"
OUTPUTS_DIR="${SCRIPT_DIR}/outputs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[KernForge]${NC} $*"; }
ok()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err() { echo -e "${RED}[✗]${NC} $*" >&2; }

# ─── Track → task mapping ───
track_to_tasks() {
    case "$1" in
        gated_delta_net|gdn)  echo "gdn";;
        fused_moe|moe)        echo "moe";;
        sparse_attention|dsa)  echo "dsa_paged";;
        all)                   echo "gdn moe dsa_paged";;
        *) err "Unknown track: $1"; exit 1;;
    esac
}

# ─── Commands ───

cmd_setup() {
    log "Setting up KernForge competition environment..."

    # 1. Python environment
    if ! command -v python3 &>/dev/null; then
        err "Python 3 not found. Please install Python 3.12+"
        exit 1
    fi

    log "Installing Python dependencies..."
    pip install -q flashinfer-bench anthropic openai modal pydantic pyyaml tqdm numpy

    # 2. Dataset
    if [ ! -d "$DATASET_DIR" ]; then
        log "Downloading mlsys26-contest dataset from HuggingFace..."
        mkdir -p "$(dirname "$DATASET_DIR")"
        git lfs install
        git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest "$DATASET_DIR"
        ok "Dataset downloaded to $DATASET_DIR"
    else
        ok "Dataset already present at $DATASET_DIR"
    fi

    # 3. Modal setup (optional)
    if command -v modal &>/dev/null; then
        log "Modal CLI found. Running modal setup..."
        modal setup 2>/dev/null || warn "Modal setup failed — you can still run locally"
        ok "Modal configured"
    else
        warn "Modal CLI not found. Install with: pip install modal && modal setup"
    fi

    # 4. API key check
    if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
        warn "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY before running."
    else
        ok "API key found"
    fi

    # 5. Verify installation
    log "Verifying KernForge installation..."
    python3 -c "
from kernforge.eval.flashinfer_eval import EvalResult, calculate_score
from kernforge.prompt.proposer_prompt import generate_proposer_prompt
from kernforge.prompt.tuner_prompt import generate_tuner_prompt
from kernforge.eval.static_analysis import analyze
print('All KernForge modules loaded successfully')
" || { err "KernForge import failed"; exit 1; }

    ok "Setup complete! Ready to compete."
    echo ""
    echo "  Next steps:"
    echo "    export ANTHROPIC_API_KEY=sk-ant-..."
    echo "    ./deploy.sh run-modal --track gated_delta_net"
}

cmd_run_modal() {
    local track="${1:-gated_delta_net}"
    local steps="${2:-25}"
    local model="${3:-claude-sonnet-4-5}"

    log "Running KernForge on Modal B200 — track: $track, steps: $steps"

    local tasks=$(track_to_tasks "$track")
    local tasks_file=$(mktemp)
    for t in $tasks; do echo "$t" >> "$tasks_file"; done

    python3 -m kernforge.main \
        --test_source mlsys26-contest \
        --agent_type iterative \
        --tasks_path "$tasks_file" \
        --gpu_name B200 \
        --gpu_architecture Blackwell \
        --api_type claude \
        --model_name "$model" \
        --total_steps "$steps" \
        --max_memory_round 5 \
        --eval_backend modal \
        --modal_gpu B200 \
        --use_static_check \
        --use_domain_knowledge

    rm -f "$tasks_file"
    ok "Run complete. Results in $OUTPUTS_DIR/"
}

cmd_run_local() {
    local track="${1:-gated_delta_net}"
    local steps="${2:-25}"
    local model="${3:-claude-sonnet-4-5}"

    log "Running KernForge on local GPU — track: $track, steps: $steps"

    local tasks=$(track_to_tasks "$track")
    local tasks_file=$(mktemp)
    for t in $tasks; do echo "$t" >> "$tasks_file"; done

    python3 -m kernforge.main \
        --test_source mlsys26-contest \
        --agent_type iterative \
        --tasks_path "$tasks_file" \
        --gpu_name B200 \
        --gpu_architecture Blackwell \
        --api_type claude \
        --model_name "$model" \
        --total_steps "$steps" \
        --max_memory_round 5 \
        --eval_backend local \
        --use_static_check \
        --use_domain_knowledge

    rm -f "$tasks_file"
    ok "Run complete. Results in $OUTPUTS_DIR/"
}

cmd_run_all_modal() {
    local steps="${1:-25}"
    log "Running ALL tracks on Modal B200..."
    for track in gated_delta_net fused_moe sparse_attention; do
        echo ""
        log "═══ Track: $track ═══"
        cmd_run_modal "$track" "$steps"
    done
    ok "All tracks complete!"
}

cmd_submit() {
    local track="${1:-}"
    if [ -z "$track" ]; then
        err "Usage: ./deploy.sh submit --track <track_name>"
        exit 1
    fi

    log "Packing submission for track: $track"

    # Find the latest output directory for this track
    local latest=$(ls -1dt "$OUTPUTS_DIR"/kernforge_*/ 2>/dev/null | head -1)
    if [ -z "$latest" ]; then
        err "No output directory found. Run the agent first."
        exit 1
    fi

    log "Using outputs from: $latest"

    # Find the best kernel for each problem in this track
    local tasks=$(track_to_tasks "$track")
    local submit_dir="$OUTPUTS_DIR/submission_${track}_$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$submit_dir"

    for task_dir in "$latest"/${tasks}_*; do
        if [ -d "$task_dir" ]; then
            local problem=$(basename "$task_dir")
            local best_kernel=$(ls -1 "$task_dir"/global_best_kernel_*.py 2>/dev/null | tail -1)
            local best_metrics=$(ls -1 "$task_dir"/global_best_metrics_*.json 2>/dev/null | tail -1)

            if [ -n "$best_kernel" ]; then
                mkdir -p "$submit_dir/$problem"
                cp "$best_kernel" "$submit_dir/$problem/kernel.py"
                [ -n "$best_metrics" ] && cp "$best_metrics" "$submit_dir/$problem/metrics.json"
                
                # Read metrics
                if [ -n "$best_metrics" ]; then
                    local speedup=$(python3 -c "import json; d=json.load(open('$best_metrics')); print(f'{d.get(\"speedup\",0):.3f}x' if d.get('correct') else 'INCORRECT')")
                    ok "$problem → $speedup"
                fi
            fi
        fi
    done

    # Pack using flashinfer-bench if available
    python3 -c "
import json, os, sys
submit_dir = '$submit_dir'
problems = [d for d in os.listdir(submit_dir) if os.path.isdir(os.path.join(submit_dir, d))]
print(f'Submission packed: {len(problems)} problems in {submit_dir}')
for p in sorted(problems):
    mf = os.path.join(submit_dir, p, 'metrics.json')
    if os.path.exists(mf):
        d = json.load(open(mf))
        status = f'{d[\"speedup\"]:.3f}x' if d.get('correct') else 'INCORRECT'
        print(f'  {p}: {status}')
" 2>/dev/null

    ok "Submission ready at: $submit_dir"
    echo ""
    echo "  Next: git add && git commit && git tag submission-v1 && git push --tags"
}

cmd_benchmark() {
    # Quick comparison: baseline vs KernForge on same problem
    local track="${1:-gated_delta_net}"
    local steps="${2:-10}"

    local ts_baseline=$(date +%s)
    local ts_kernforge=$((ts_baseline + 1))
    local baseline_dir="$OUTPUTS_DIR/benchmark_baseline_${ts_baseline}"
    local kernforge_dir="$OUTPUTS_DIR/benchmark_kernforge_${ts_kernforge}"

    log "Benchmark: baseline vs KernForge on $track ($steps steps each)"
    echo ""

    # Run baseline mode (no enhancements)
    log "Running BASELINE (no domain knowledge, no static analysis)..."
    python3 -m kernforge.main \
        --test_source mlsys26-contest \
        --agent_type iterative \
        --tasks_path <(echo "$(track_to_tasks "$track")") \
        --total_steps "$steps" \
        --eval_backend modal --modal_gpu B200 \
        --no_enhancements \
        --save_path "$baseline_dir" \
        2>&1 | tee /tmp/kernforge_baseline.log

    echo ""

    # Run KernForge mode
    log "Running KERNFORGE (full enhancements)..."
    python3 -m kernforge.main \
        --test_source mlsys26-contest \
        --agent_type iterative \
        --tasks_path <(echo "$(track_to_tasks "$track")") \
        --total_steps "$steps" \
        --eval_backend modal --modal_gpu B200 \
        --use_static_check --use_domain_knowledge \
        --save_path "$kernforge_dir" \
        2>&1 | tee /tmp/kernforge_enhanced.log

    echo ""
    log "Structured comparison:"

    # Structured comparison via Python
    python3 - "$baseline_dir" "$kernforge_dir" <<'PYEOF'
import json, os, sys, statistics

baseline_dir = sys.argv[1]
kernforge_dir = sys.argv[2]

def load_metrics(run_dir):
    """Load all metrics.json files from a run directory, keyed by problem name."""
    results = {}
    if not os.path.isdir(run_dir):
        return results
    for root, dirs, files in os.walk(run_dir):
        for f in files:
            if f.startswith("global_best_metrics") and f.endswith(".json"):
                problem = os.path.basename(root)
                try:
                    with open(os.path.join(root, f)) as fh:
                        data = json.load(fh)
                    # Keep the best (highest speedup) if multiple
                    existing = results.get(problem)
                    if existing is None or data.get("speedup", 0) > existing.get("speedup", 0):
                        results[problem] = data
                except (json.JSONDecodeError, OSError):
                    pass
    return results

baseline = load_metrics(baseline_dir)
kernforge = load_metrics(kernforge_dir)
all_problems = sorted(set(list(baseline.keys()) + list(kernforge.keys())))

if not all_problems:
    print("  No metrics found. Ensure both runs completed successfully.")
    sys.exit(0)

# Header
print(f"  {'Problem':<40s} {'Baseline':>12s} {'KernForge':>12s} {'Delta':>10s}")
print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*10}")

deltas = []
wins = 0
total = 0

for prob in all_problems:
    b = baseline.get(prob, {})
    k = kernforge.get(prob, {})
    b_speed = b.get("speedup") if b.get("correct") else None
    k_speed = k.get("speedup") if k.get("correct") else None

    b_str = f"{b_speed:.3f}x" if b_speed is not None else ("WRONG" if b else "N/A")
    k_str = f"{k_speed:.3f}x" if k_speed is not None else ("WRONG" if k else "N/A")

    delta_str = ""
    if b_speed is not None and k_speed is not None:
        delta = k_speed - b_speed
        deltas.append(delta)
        total += 1
        if delta > 0:
            wins += 1
        delta_str = f"{delta:+.3f}x"

    print(f"  {prob:<40s} {b_str:>12s} {k_str:>12s} {delta_str:>10s}")

# Aggregate stats
print(f"\n  {'='*76}")
if deltas:
    print(f"  Mean delta:   {statistics.mean(deltas):+.3f}x")
    print(f"  Median delta: {statistics.median(deltas):+.3f}x")
    print(f"  Win rate:     {wins}/{total} ({100*wins/total:.0f}%)")
else:
    print("  No comparable results (both runs need correct solutions on same problems)")
PYEOF
}

# ─── CLI router ───
case "${1:-help}" in
    setup)
        cmd_setup
        ;;
    run-modal)
        shift
        track=""
        steps=25
        model="claude-sonnet-4-5"
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --track) track="$2"; shift 2;;
                --steps) steps="$2"; shift 2;;
                --model) model="$2"; shift 2;;
                *) track="$1"; shift;;
            esac
        done
        cmd_run_modal "${track:-gated_delta_net}" "$steps" "$model"
        ;;
    run-local)
        shift
        track=""
        steps=25
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --track) track="$2"; shift 2;;
                --steps) steps="$2"; shift 2;;
                *) track="$1"; shift;;
            esac
        done
        cmd_run_local "${track:-gated_delta_net}" "$steps"
        ;;
    run-all-modal)
        shift
        cmd_run_all_modal "${1:-25}"
        ;;
    submit)
        shift
        track=""
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --track) track="$2"; shift 2;;
                *) track="$1"; shift;;
            esac
        done
        cmd_submit "$track"
        ;;
    benchmark)
        shift
        cmd_benchmark "${1:-gated_delta_net}" "${2:-10}"
        ;;
    help|--help|-h|*)
        echo "KernForge Competition Deploy"
        echo ""
        echo "Commands:"
        echo "  setup                         Install deps, download dataset, configure Modal"
        echo "  run-modal  --track <track>    Run on Modal B200 (no local GPU needed)"
        echo "  run-local  --track <track>    Run on local GPU"
        echo "  run-all-modal                 Run all 3 tracks on Modal"
        echo "  submit     --track <track>    Pack best kernels for submission"
        echo "  benchmark  [track] [steps]    A/B test: baseline vs KernForge"
        echo ""
        echo "Tracks: gated_delta_net | fused_moe | sparse_attention"
        echo ""
        echo "Examples:"
        echo "  ./deploy.sh setup"
        echo "  ./deploy.sh run-modal --track gated_delta_net --steps 25"
        echo "  ./deploy.sh benchmark gated_delta_net 10"
        echo "  ./deploy.sh submit --track gated_delta_net"
        ;;
esac
