#!/usr/bin/env python3
"""Generate README SVG charts from the committed, explicitly labeled benchmark logs."""
from pathlib import Path
import html
import re

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONFIGS = [
    ("baseline", 0, "baseline", "bench_baseline.txt"),
    ("k=2 causal", 2, "causal", "bench_dflash_k2_causal.txt"),
    ("k=2 bidirectional", 2, "bidirectional", "bench_dflash_k2_bidirectional.txt"),
    ("k=4 causal", 4, "causal", "bench_dflash_k4_causal.txt"),
    ("k=4 bidirectional", 4, "bidirectional", "bench_dflash_k4_bidirectional.txt"),
    ("k=8 causal", 8, "causal", "bench_dflash_k8_causal.txt"),
    ("k=8 bidirectional", 8, "bidirectional", "bench_dflash_k8_bidirectional.txt"),
    ("k=15 causal", 15, "causal", "bench_dflash_k15_causal.txt"),
]


def parse(label, k, mode, filename):
    path = RESULTS / filename
    text = path.read_text(encoding="utf-8")
    mean = float(re.search(r"overall:.*?= ([\d.]+) tok/s", text).group(1))
    median = float(re.search(r"per-request median ([\d.]+) tok/s", text).group(1))
    acc = re.search(r"acceptance rate: ([\d.]+)%", text)
    positions = {
        int(i): int(v)
        for i, v in re.findall(r'position="(\d+)"\} = (\d+)', text)
    }
    return {"label": label, "k": k, "mode": mode, "mean": mean, "median": median,
            "acceptance": float(acc.group(1)) if acc else None, "positions": positions}


def throughput_svg(rows):
    width, left, top, row_h = 760, 210, 72, 39
    height = top + row_h * len(rows) + 42
    vmax = max(r["mean"] for r in rows) * 1.16
    scale = (width - left - 120) / vmax
    colors = {"baseline": "#64748b", "causal": "#ea580c", "bidirectional": "#2563eb"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
           '<title>Qwen3-8B decode throughput on AWS Trainium2</title>',
           '<desc>Baseline and DFlash causal and bidirectional query-block mask throughput.</desc>',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{width/2}" y="27" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="700">Qwen3-8B decode throughput on AWS Trainium2</text>',
           f'<text x="{width/2}" y="49" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#475569">six requests per configuration; one request at a time; 128 completion tokens each</text>']
    for n, row in enumerate(rows):
        y = top + n * row_h
        bar = row["mean"] * scale
        color = colors[row["mode"]]
        out += [f'<text x="{left-10}" y="{y+21}" text-anchor="end" font-family="sans-serif" font-size="13">{html.escape(row["label"])}</text>',
                f'<rect x="{left}" y="{y+5}" width="{bar:.1f}" height="23" rx="2" fill="{color}"/>',
                f'<text x="{left+bar+8:.1f}" y="{y+21}" font-family="sans-serif" font-size="13" font-weight="700">{row["mean"]:.1f} tok/s</text>']
    out.append('</svg>')
    return "\n".join(out) + "\n"


def acceptance_svg(rows):
    causal = next(r for r in rows if r["k"] == 8 and r["mode"] == "causal")
    bidir = next(r for r in rows if r["k"] == 8 and r["mode"] == "bidirectional")
    width, height, left, top, plot_w, plot_h = 760, 350, 65, 65, 565, 220
    maximum = max(bidir["positions"].values())
    colors = {"causal": "#ea580c", "bidirectional": "#2563eb"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
           '<title>Accepted DFlash tokens by draft position at k=8</title>',
           '<desc>Counts for causal and bidirectional query-block attention.</desc>',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="700">Accepted DFlash tokens by position (k=8)</text>']
    for tick in range(0, 201, 50):
        y = top + plot_h * (1 - tick / maximum)
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>',
                f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#475569">{tick}</text>']
    for pos in range(8):
        x = left + plot_w * pos / 7
        out.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{pos}</text>')
    for row in (causal, bidir):
        points = []
        for pos in range(8):
            x = left + plot_w * pos / 7
            y = top + plot_h * (1 - row["positions"][pos] / maximum)
            points.append(f'{x:.1f},{y:.1f}')
        out.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[row["mode"]]}" stroke-width="3"/>')
        for point in points:
            x, y = point.split(',')
            out.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{colors[row["mode"]]}"/>')
    for n, mode in enumerate(("causal", "bidirectional")):
        y = 84 + n * 24
        out += [f'<line x1="650" y1="{y}" x2="676" y2="{y}" stroke="{colors[mode]}" stroke-width="3"/>',
                f'<text x="684" y="{y+4}" font-family="sans-serif" font-size="12">{mode}</text>']
    out += [f'<text x="{left+plot_w/2}" y="{height-16}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#475569">draft position (0 is the first proposal)</text>', '</svg>']
    return "\n".join(out) + "\n"


def main():
    rows = [parse(*item) for item in CONFIGS]
    (RESULTS / "throughput.svg").write_text(throughput_svg(rows), encoding="utf-8")
    (RESULTS / "acceptance.svg").write_text(acceptance_svg(rows), encoding="utf-8")
    baseline = rows[0]["mean"]
    print("| configuration | tok/s | speedup | acceptance |")
    print("|---|---:|---:|---:|")
    for row in rows:
        acceptance = "n/a" if row["acceptance"] is None else f'{row["acceptance"]:.1f}%'
        print(f'| {row["label"]} | {row["mean"]:.1f} | {row["mean"]/baseline:.2f}x | {acceptance} |')


if __name__ == "__main__":
    main()
