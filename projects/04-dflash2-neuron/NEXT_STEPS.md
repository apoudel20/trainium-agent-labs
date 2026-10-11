# Next experiments — ranked by ROI (grounded in measured data)

All speedups below are vs the no-speculation baseline (51.7 tok/s), measured on
one trn2 chip (seat-40), TP=2, greedy, 128 new tokens. Current best:
**bidirectional k=4/k=8 ≈ 88–89 tok/s (1.70–1.73×)**.

## Why "just raise the draft-token count" is NOT the high-ROI lever

Accepted tokens by draft position (real `k=8` bidirectional run):

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---------:|---:|---:|--:|--:|--:|--:|--:|--:|
| accepted | 204 | 118 | 64 | 33 | 17 | 7 | 5 | 5 |

Acceptance collapses after position ~4 and is ~flat (5–7) by positions 5–7.
Every extra drafted token still costs a target verification, so adding positions
8–14 (i.e. k=15) adds almost no accepted tokens while nearly doubling verify
cost. This already happened in the causal sweep: **k=15 (67.3) was slower than
k=8 (69.0)**. The draft is a tiny 5-layer block-16 model — it cannot predict far
ahead. The ceiling is in the *draft quality*, not the token count.

## ROI ranking

### 1. (HIGH) Real DFlash2 draft with the candidate selector
The deployed 8B draft is unary DFlash (greedy top-1 per position). The actual
DFlash2 draft reranks **top-16 candidates per position** via a
predecessor/successor transition score and walks one coherent path — explicitly
designed to stop acceptance decaying across the block. This lifts the *whole*
per-position curve, which is the measured bottleneck, and is what makes longer
blocks finally pay off. Components already implemented + verified against real
weights (`dflash2_27b/dflash2_components.py`, `DFLASH2_COMPONENTS: PASS`).
Remaining: wire into a Neuron proposer.

### 2. (HIGH, highest payoff + the paper's goal) Qwen3.8-27B + DFlash2 end-to-end
The paper target. Both Qwen3.5 attention variants (GatedDeltaNet/SSM + full)
validated against the real 27B weights; DFlash2 conv+selector validated. Remaining
is the Neuron assembly of the 64-layer hybrid decoder (dual KV + SSM-state cache)
and serving — larger effort, but every building block is proven in isolation.

### 3. (LOW payoff, LOW cost) Confirm the k plateau: bidirectional k=12 / k=15
We never ran k=15 *with* the bidirectional patch (only causal). The position
curve predicts a plateau or slight regression, not a win. Worth running only to
*confirm the ceiling* — frame as verification, not an expected speedup. Cheap.

### Also worth a cheap check
- **k=5 / k=6 bidirectional**: fill the gap between the k=4 and k=8 tie to find
  the exact throughput peak (both are ~88–89 tok/s; the true optimum is in between).

## Status note
These require the Trainium pod. As of 2026-10-11 the EKS/kubectl credentials for
seat-40 are expired, so no new on-device runs can be made until access is
refreshed. All results above are already-measured and committed.
