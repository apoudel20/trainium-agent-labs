# DFlash speculative decoding on AWS Trainium2

**Added DFlash speculative decoding to vLLM-Neuron for a Qwen3-8B target. With four draft
positions per step, bidirectional draft attention raised acceptance from 20.1% to 32.7% and
improved decode throughput from 1.34x to 1.70x over the autoregressive baseline.**

[DFlash](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) is a block-diffusion draft model
for speculative decoding. Its five-layer draft network reads intermediate hidden states from
Qwen3-8B, proposes several next tokens in one pass, and lets the target model verify those
proposals together. vLLM supported this model on GPU, but the installed Neuron backend could
not load it; the first failure was:

```
AttributeError: type object 'DFlashQwen3ForCausalLM' has no attribute 'from_configs'
```

This project was developed by Aayush Poudel during the NYU × Annapurna Labs **Hack the Chip**
hackathon in October 2026. It is part of a fork of the organizers'
[`yahavb/trainium-agent-labs`](https://github.com/yahavb/trainium-agent-labs) repository.

[Live results](https://apoudel20.github.io/trainium-agent-labs/) ·
[Raw benchmark logs](results/) ·
[DFlash model card](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) ·
[DFlash paper](https://arxiv.org/abs/2602.06036) ·
[vLLM-Neuron](https://github.com/vllm-project/vllm-neuron)

![Throughput comparison](results/throughput.svg)

## Results

The table reports a small, controlled benchmark: one request at a time, greedy decoding,
128 generated tokens, three prompts repeated twice (six requests per configuration). `k` is
the number of draft positions proposed per target step. Draft acceptance is accepted draft
tokens divided by proposed draft tokens. Throughput (`tok/s`) is total completion tokens
divided by total request wall time across the six requests.

All rows used one AWS Trainium2 chip, tensor parallelism over two NeuronCores (`TP=2`),
`max_model_len=256`, `max_num_seqs=1`, and `max_num_batched_tokens=256`. Only the draft
count and query-block mask changed.

| Configuration | Mean tok/s | Per-request median | Per-request range | Speedup | Draft acceptance |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B, autoregressive baseline | 51.7 | 51.7 | 51.6–51.7 | 1.00x | n/a |
| DFlash k=2, causal | 64.7 | 65.0 | 61.4–69.3 | 1.25x | 33.1% |
| DFlash k=2, bidirectional | 73.8 | 76.0 | 69.7–76.1 | 1.43x | 44.0% |
| DFlash k=4, causal | 69.3 | 66.5 | 64.8–78.0 | 1.34x | 20.1% |
| **DFlash k=4, bidirectional** | **88.1** | **85.9** | **78.4–104.1** | **1.70x** | **32.7%** |
| DFlash k=8, causal | 69.0 | 68.2 | 61.6–77.5 | 1.33x | 11.1% |
| DFlash k=8, bidirectional | 89.4 | 89.2 | 87.3–94.6 | 1.73x | 17.8% |
| DFlash k=15, causal only | 67.3 | 65.7 | 63.4–74.4 | 1.30x | 6.0% |

The consistent headline uses k=4 because its throughput and mask comparison come from the
same draft count: acceptance rose from 20.1% to 32.7%, while throughput rose from 69.3 to
88.1 tok/s. The k=8 bidirectional run reached the highest observed throughput, 89.4 tok/s
(1.73x), but it is a different configuration.

At k=8, bidirectional attention increased accepted counts most in absolute terms at positions
0 and 1 (+34 and +36); the largest relative gains were at positions 1 and 2 (+44% and +31%).
Positions 4–6 were unchanged or slightly lower, while position 7 increased by three. This is
consistent with the causal mask withholding useful intra-block context, but the benchmark
alone does not establish that mechanism.

The sample is deliberately small, so these numbers should be read as hackathon measurements,
not production performance estimates. Each row is traceable to a committed log in
[`results/`](results/). The baseline is the 2026-10-10 22:43 UTC run recorded in
[`bench_baseline.txt`](results/bench_baseline.txt).

![Draft acceptance by position](results/acceptance.svg)

## Reproduce

Tested with this container image and SDK stack:

```
public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:
  0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04
```

The host must expose one Trainium2 chip. Clone this branch, enter this directory, and install
the patch into the active Python environment:

```bash
git clone --branch dflash2-neuron https://github.com/apoudel20/trainium-agent-labs.git
cd trainium-agent-labs/projects/04-dflash2-neuron

bash apply.sh                         # bidirectional build used for headline results
bash serve_dflash2.sh                 # defaults to k=4 on port 8000
bash tools/wait_log.sh                # returns on ready/failure; 60-second maximum per call
bash run_bench.sh 4                   # writes results/bench_dflash_k4_bidirectional.txt
```

To reproduce a causal row instead:

```bash
bash apply.sh --causal
DFLASH_NUM_SPEC=4 bash serve_dflash2.sh
bash tools/wait_log.sh
bash run_bench.sh 4                   # writes results/bench_dflash_k4_causal.txt
```

Return to the default build with `bash apply.sh --bidirectional`. `apply.sh --check` reports
which base and mask patches are installed without changing files.

For the baseline:

```bash
bash serve_baseline.sh
bash tools/wait_log.sh /tmp/baseline.log
bash run_bench.sh 0
```

The first start compiles graphs for the chip; later starts reuse the Neuron compile cache.
No model weights are included in this repository.

## Implementation

The default installer copies the overlay, applies [`vllm_neuron-dflash2.patch`](vllm_neuron-dflash2.patch),
and then applies [`bidirectional_attention.patch`](bidirectional_attention.patch). The causal
switch reverses only the mask patch.

| File | Change |
|---|---|
| `overlay/vllm_neuron/model/qwen3/dflash_draft.py` | Five-layer Neuron draft model, target hidden-state projection, context KV-cache population, and draft query block |
| `overlay/vllm_neuron/vllm/spec_decode/dflash2.py` | Standalone proposer that loads the draft checkpoint and shares Qwen3-8B embedding/LM-head weights |
| `vllm_neuron-dflash2.patch` | Qwen3 target hidden-state capture, DFlash checkpoint loading, registry entry, and runner dispatch |
| `bidirectional_attention.patch` | Optional all-ones active mask within the draft query block; enabled by default |

For k=4, one draft step constructs `[next token, MASK, MASK, MASK, MASK]`. The draft writes
context K/V from five target hidden states, evaluates the five-position query block, and
returns four proposals. The target verifies those proposals; accepted output therefore follows
the target's greedy choices, subject to bf16 numerical differences between single-token and
multi-token target kernels.

**Naming note.** The served checkpoint is the original DFlash model
`z-lab/Qwen3-8B-DFlash-b16`, not the later DFlash 2 architecture. Some implementation paths
and class names retain `dflash2` from the original hackathon prototype. The separate
[`dflash2_27b/`](dflash2_27b/) directory contains exploratory component checks for the later
`z-lab/Qwen3.8-27B-DFlash2` checkpoint; it does not provide end-to-end serving.

## Related work

This is an independent hackathon implementation, not the first DFlash work for Trainium.
Upstream vLLM-Neuron has open, more complete proposals:

- [vllm-project/vllm-neuron#47](https://github.com/vllm-project/vllm-neuron/pull/47) — DFlash framework support.
- [vllm-project/vllm-neuron#50](https://github.com/vllm-project/vllm-neuron/pull/50) — Qwen3 target support and measurements.

Those upstream changes include broader validation and bidirectional block handling. This project
is retained as the reproducible hackathon result and its small Qwen3-8B mask/draft-count sweep.

## Limitations

- Batch size/concurrency was one; batching and longer contexts were not measured.
- Bidirectional k=15 was not measured. The only k=15 result is causal.
- Greedy text from the k=4 speculative run matched the baseline for roughly 95 tokens, then
  diverged on a near-tie. Both continuations were coherent; no claim of bit-exact output is made.
- If the draft query block reaches past the request's allocated KV-cache blocks, the draft clamps
  the block index to the last allocated block. This can lower draft acceptance near that
  boundary; the target's cache is not affected, and the target still verifies every draft.

## Generated artifacts

`make_charts.py` regenerates the standalone SVGs used by this README.
`make_dashboard.py` reads the same committed logs and writes the GitHub Pages source to
`docs/index.html`. The Pages source is `docs/` on branch `dflash2-neuron`.
