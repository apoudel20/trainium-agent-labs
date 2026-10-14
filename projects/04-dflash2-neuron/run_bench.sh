#!/usr/bin/env bash
# run_bench.sh <num_draft_tokens|0 for baseline> — save one benchmark under results/.
set -euo pipefail
K=${1:?usage: run_bench.sh <num_draft_tokens, 0 = baseline>}
HERE="$(cd "$(dirname "$0")" && pwd)"
SP="${VLLM_SITE_PACKAGES:-$(python3 -c 'import os, vllm_neuron; print(os.path.dirname(os.path.dirname(vllm_neuron.__file__)))' 2>/dev/null)}"
mkdir -p "$HERE/results"

if [ "$K" = 0 ]; then
    MODE=baseline
    PATCH_STATE=not-applicable
    OUT="$HERE/results/bench_baseline.txt"
    SERVER="Qwen/Qwen3-8B; speculative decoding disabled"
else
    if grep -q '"bidirectional_active": True' "$SP/vllm_neuron/model/qwen3/dflash_draft.py" 2>/dev/null; then
        MODE=bidirectional
        PATCH_STATE=applied
    else
        MODE=causal
        PATCH_STATE=not-applied
    fi
    OUT="$HERE/results/bench_dflash_k${K}_${MODE}.txt"
    SERVER="Qwen/Qwen3-8B + z-lab/Qwen3-8B-DFlash-b16; ${K} draft tokens"
fi

{
    echo "# command: python3 bench_dflash2.py --max-tokens 128 --runs 2 --mask-mode $MODE"
    echo "# server: $SERVER"
    echo "# mask_mode=$MODE bidirectional_attention_patch=$PATCH_STATE"
    echo "# hardware: one AWS Trainium2 chip, tensor parallel over 2 NeuronCores; $(date -u '+%Y-%m-%d %H:%M UTC')"
    python3 "$HERE/bench_dflash2.py" --max-tokens 128 --runs 2 --mask-mode "$MODE"
} | tee "$OUT"
echo "saved: $OUT"
