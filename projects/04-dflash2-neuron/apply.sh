#!/usr/bin/env bash
# Install DFlash speculative decoding support into the active Python environment's vllm_neuron.
#
#   bash apply.sh             # bidirectional query-block mask (default/headline build)
#   bash apply.sh --causal    # original causal MVP for causal benchmark rows
#   bash apply.sh --check     # report installed state without changing files
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SP="${VLLM_SITE_PACKAGES:-$(python3 -c 'import os, vllm_neuron; print(os.path.dirname(os.path.dirname(vllm_neuron.__file__)))' 2>/dev/null)}"
BASE_PATCH="$HERE/vllm_neuron-dflash2.patch"
MASK_PATCH="$HERE/bidirectional_attention.patch"
MODE=bidirectional
CHECK_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --causal) MODE=causal ;;
        --bidirectional) MODE=bidirectional ;;
        --check) CHECK_ONLY=1 ;;
        *) echo "usage: bash apply.sh [--bidirectional|--causal] [--check]" >&2; exit 2 ;;
    esac
done

[ -d "$SP/vllm_neuron" ] || { echo "vllm_neuron not found (site-packages: $SP)" >&2; exit 1; }
patch_applied() { patch -p1 -d "$SP" --dry-run -R -s -f < "$1" >/dev/null 2>&1; }

echo "site-packages: $SP"
if [ "$CHECK_ONLY" = 1 ]; then
    patch_applied "$BASE_PATCH" && echo "DFlash base patch: applied" || echo "DFlash base patch: not applied"
    patch_applied "$MASK_PATCH" && echo "query-block mask: bidirectional" || echo "query-block mask: causal/not installed"
    exit 0
fi
echo "requested mask mode: $MODE"

# Normalize an existing install to the causal base before copying the causal overlay.
# This keeps repeated calls and mode switches deterministic.
if patch_applied "$MASK_PATCH"; then
    patch -p1 -d "$SP" -R -s -f < "$MASK_PATCH"
fi

cp -v "$HERE/overlay/vllm_neuron/model/qwen3/dflash_draft.py" "$SP/vllm_neuron/model/qwen3/"
cp -v "$HERE/overlay/vllm_neuron/vllm/spec_decode/dflash2.py" "$SP/vllm_neuron/vllm/spec_decode/"

if patch_applied "$BASE_PATCH"; then
    echo "DFlash base patch: already applied"
else
    patch -p1 -d "$SP" --dry-run -f < "$BASE_PATCH"
    patch -p1 -d "$SP" -f < "$BASE_PATCH"
fi

if [ "$MODE" = bidirectional ]; then
    patch -p1 -d "$SP" --dry-run -f < "$MASK_PATCH"
    patch -p1 -d "$SP" -f < "$MASK_PATCH"
fi

python3 -m py_compile \
    "$SP/vllm_neuron/model/qwen3/dflash_draft.py" \
    "$SP/vllm_neuron/vllm/spec_decode/dflash2.py" \
    "$SP/vllm_neuron/model/qwen3/model.py" \
    "$SP/vllm_neuron/model/registry.py" \
    "$SP/vllm_neuron/vllm/worker/neuron_model_runner.py" \
    "$SP/vllm_neuron/functional/attention/attention_decode_mask.py"
echo "DFlash installed with $MODE query-block attention."
echo "Start it with: bash $HERE/serve_dflash2.sh"
