# Follow-up work

The completed 8B experiment serves `Qwen/Qwen3-8B` with the original
`z-lab/Qwen3-8B-DFlash-b16` draft on one AWS Trainium2 chip. At k=4, bidirectional
query-block attention measured 88.1 tok/s (1.70x over the 51.7 tok/s baseline) and 32.7%
draft-token acceptance.

## Draft count

Accepted tokens by draft position in the k=8 bidirectional run were:

| Position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accepted | 204 | 118 | 64 | 33 | 17 | 7 | 5 | 5 |

Acceptance falls sharply across the block. The causal k=15 run reached 67.3 tok/s, below the
causal k=8 run at 69.0 tok/s. Bidirectional k=15 was not measured, so the available evidence
does not establish its result. Measuring k=5, k=6, and bidirectional k=15 would locate the
throughput peak more precisely.

## Later DFlash 2 checkpoint

The served 8B checkpoint is DFlash, not DFlash 2. The later
[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
draft adds dynamic convolution and candidate selection. The component checks in
[`dflash2_27b/`](dflash2_27b/) load released weights and validate those mechanisms in
isolation; they do not provide an end-to-end Neuron server.

The matching `Qwen/Qwen3.8-27B` target reports `model_type: qwen3_5_text`. In other words,
Qwen3.8 uses the Qwen3.5 hybrid text architecture: interleaved linear-attention/SSM and
full-attention layers. The installed Neuron backend implements dense Qwen3, not that hybrid
64-layer target. End-to-end serving therefore requires a Neuron implementation of the hybrid
decoder and its SSM state cache before the DFlash 2 components can be connected.

## Other measurements

The current benchmark covers one request at a time and a 256-token model length. Useful
follow-ups are:

1. Repeat the comparison at longer context lengths.
2. Measure concurrency above one; speculative verification cost can scale differently with
   batch size.
3. Profile target verification and draft execution separately with Neuron Explorer.
4. Compare this hackathon implementation with the broader upstream work in
   [vllm-neuron#47](https://github.com/vllm-project/vllm-neuron/pull/47) and
   [vllm-neuron#50](https://github.com/vllm-project/vllm-neuron/pull/50).
