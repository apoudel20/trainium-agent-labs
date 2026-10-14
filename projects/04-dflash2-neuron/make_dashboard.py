#!/usr/bin/env python3
"""Generate docs/index.html and README SVGs from committed benchmark logs."""
from pathlib import Path
import html
import re

import make_charts

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DOCS = HERE.parents[1] / "docs"
REPO = "https://github.com/apoudel20/trainium-agent-labs"
PROJECT = f"{REPO}/tree/dflash2-neuron/projects/04-dflash2-neuron"


def run_range(filename):
    text = (RESULTS / filename).read_text(encoding="utf-8")
    values = [float(v) for v in re.findall(r"run \d+ prompt \d+:.*?= ([\d.]+) tok/s", text)]
    return f"{min(values):.1f}–{max(values):.1f}"


def main():
    make_charts.main()
    rows = [make_charts.parse(*item) for item in make_charts.CONFIGS]
    baseline = rows[0]["mean"]
    ranges = {item[3]: run_range(item[3]) for item in make_charts.CONFIGS}
    throughput = (RESULTS / "throughput.svg").read_text(encoding="utf-8")
    acceptance = (RESULTS / "acceptance.svg").read_text(encoding="utf-8")

    table_rows = []
    for row, item in zip(rows, make_charts.CONFIGS):
        filename = item[3]
        acceptance_value = "n/a" if row["acceptance"] is None else f'{row["acceptance"]:.1f}%'
        strong = row["k"] == 4 and row["mode"] == "bidirectional"
        cells = [html.escape(row["label"]), f'{row["mean"]:.1f}', f'{row["median"]:.1f}',
                 ranges[filename], f'{row["mean"]/baseline:.2f}x', acceptance_value]
        row_class = ' class="highlight"' if strong else ""
        table_rows.append(f"<tr{row_class}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    page = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DFlash speculative decoding on AWS Trainium2 — Aayush Poudel</title>
  <meta name="description" content="A vLLM-Neuron port of DFlash for Qwen3-8B, measured at 1.70x decode throughput with four draft tokens on AWS Trainium2.">
  <meta property="og:title" content="DFlash speculative decoding on AWS Trainium2">
  <meta property="og:description" content="Qwen3-8B measured 1.70x decode throughput; a bidirectional draft mask raised acceptance from 20.1% to 32.7% at k=4.">
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://apoudel20.github.io/trainium-agent-labs/">
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#526071; --line:#dbe2ea; --blue:#1d4ed8; --bg:#f8fafc; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--bg); font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }}
    main {{ max-width:940px; margin:auto; padding:42px 22px 60px; }}
    h1 {{ font-size:clamp(2rem,5vw,3.2rem); line-height:1.08; margin:0 0 10px; letter-spacing:-.03em; }}
    h2 {{ margin:40px 0 12px; font-size:1.45rem; }}
    h3 {{ margin:24px 0 8px; }}
    p {{ max-width:78ch; }}
    .byline,.note {{ color:var(--muted); }}
    .links {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin:18px 0 28px; }}
    a {{ color:var(--blue); text-decoration-thickness:1px; text-underline-offset:3px; }}
    .summary {{ background:#fff; border:1px solid var(--line); border-left:5px solid var(--blue); border-radius:8px; padding:18px 20px; font-size:1.08rem; }}
    .chart {{ overflow-x:auto; background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px; margin:18px 0; }}
    .chart svg {{ display:block; max-width:100%; height:auto; margin:auto; }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ border-collapse:collapse; width:100%; min-width:760px; background:#fff; font-size:.92rem; }}
    th,td {{ border:1px solid var(--line); padding:9px 10px; text-align:right; white-space:nowrap; }}
    th:first-child,td:first-child {{ text-align:left; }}
    thead th {{ background:#eef2f7; }}
    tr.highlight td {{ background:#eff6ff; font-weight:700; }}
    code {{ background:#e9eef5; padding:.1em .3em; border-radius:4px; }}
    footer {{ margin-top:48px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); }}
  </style>
</head>
<body><main>
  <h1>DFlash speculative decoding on AWS Trainium2</h1>
  <p class="byline">Aayush Poudel · NYU × Annapurna Labs “Hack the Chip” hackathon · October 2026</p>
  <nav class="links" aria-label="Project links">
    <a href="{PROJECT}">Code and reproduction</a>
    <a href="{PROJECT}/results">Raw results</a>
    <a href="{REPO}/blob/dflash2-neuron/projects/04-dflash2-neuron/NEXT_STEPS.md">Follow-up work</a>
    <a href="https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16">DFlash model</a>
    <a href="https://arxiv.org/abs/2602.06036">Paper</a>
    <a href="https://github.com/vllm-project/vllm-neuron">vLLM-Neuron</a>
  </nav>

  <p>DFlash is a block-diffusion draft model for speculative decoding: a small model proposes
  several future tokens in one pass, and the Qwen3-8B target verifies those proposals together.
  vLLM supported the model on GPU, but the installed Neuron backend failed while loading it
  (<code>DFlashQwen3ForCausalLM</code> had no <code>from_configs</code> method). This work added
  the Neuron draft model, proposer, Qwen3 hidden-state export, weight loading, and runner dispatch.</p>

  <div class="summary"><strong>Measured result at k=4:</strong> bidirectional query-block
  attention raised draft acceptance from <strong>20.1% to 32.7%</strong> and increased Qwen3-8B
  decode throughput from <strong>1.34x to 1.70x</strong> over the autoregressive baseline.</div>

  <h2>Benchmark</h2>
  <p>This is a small hackathon benchmark: one request at a time, greedy decoding, three prompts
  repeated twice, and 128 generated tokens per request. Throughput is total completion tokens
  divided by total request wall time across those six requests. <em>k</em> is the number of draft
  positions proposed per target step. Draft acceptance is accepted draft tokens divided by all
  proposed draft tokens.</p>
  <p>Every row used one Trainium2 chip with tensor parallelism over two NeuronCores (TP=2), a
  256-token model length, and the same vLLM settings. The sample is too small to characterize
  production performance; the per-request median and range show the observed spread.</p>

  <div class="chart">{throughput}</div>
  <div class="table-wrap"><table>
    <thead><tr><th>Configuration</th><th>Mean tok/s</th><th>Median</th><th>Range</th><th>Speedup</th><th>Acceptance</th></tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table></div>

  <h2>Effect of the query-block mask</h2>
  <p>The DFlash draft was trained with bidirectional attention inside its proposed-token block.
  The first Neuron version reused the causal decode mask. Replacing that active-block mask with
  an all-ones mask improved k=4 acceptance from 20.1% to 32.7% and throughput from 69.3 to
  88.1 tok/s.</p>
  <div class="chart">{acceptance}</div>
  <p>At k=8, the largest absolute increases in accepted tokens were at positions 0 and 1
  (+34 and +36); the largest relative increases were at positions 1 and 2 (+44% and +31%).
  Positions 4–6 were unchanged or slightly lower. Bidirectional access to the proposed-token
  block is a likely explanation, consistent with the training configuration, but these counts
  alone do not establish causality.</p>

  <h2>Reproducibility and scope</h2>
  <p>By default the installer applies the bidirectional mask used for the headline result.
  <code>bash apply.sh --causal</code> reproduces the earlier causal build. Each committed result
  log records its mask mode and patch state. The exact commands, tested container image, raw logs,
  implementation notes, and known limitations are in the
  <a href="{PROJECT}">project README</a>.</p>
  <p>The served checkpoint is <strong>DFlash</strong>
  (<code>z-lab/Qwen3-8B-DFlash-b16</code>), not the later DFlash 2 architecture. A separate set of
  component checks for the later Qwen3.8-27B DFlash 2 model is documented as follow-up work; it
  is not an end-to-end server.</p>
  <p>The speculative k=4 output matched the baseline's greedy text for roughly 95 tokens and then
  diverged on a near-tie, consistent with bf16 differences between single-token and multi-token
  target kernels. Accepted drafts are checked by the target, but no bit-exact output claim is made.</p>

  <h2>Context and related work</h2>
  <p>This repository is a fork of the organizers'
  <a href="https://github.com/yahavb/trainium-agent-labs">trainium-agent-labs</a> workshop project.
  The DFlash port was developed independently during the hackathon. Broader upstream DFlash work
  exists in <a href="https://github.com/vllm-project/vllm-neuron/pull/47">vllm-neuron #47</a> and
  <a href="https://github.com/vllm-project/vllm-neuron/pull/50">#50</a>.</p>

  <footer><a href="{PROJECT}">View the code and full reproduction notes on GitHub</a>.</footer>
</main></body></html>'''

    DOCS.mkdir(exist_ok=True)
    output = DOCS / "index.html"
    output.write_text(page, encoding="utf-8")
    print(f"Wrote {output.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
