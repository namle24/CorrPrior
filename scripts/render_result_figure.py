"""Render the documented PSNR effects as a deterministic, dependency-free SVG.

This plots reported aggregates; it does not recompute confidence intervals or
run inference. Use --check in CI to detect stale plots after editing the CSV.
"""
import argparse
import csv
from html import escape
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/assets/psnr_effects.csv"
OUTPUT = ROOT / "docs/assets/psnr_effects.svg"


def read_effects(path=DATA):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("delta_psnr_db", "ci_low_db", "ci_high_db"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f"Non-finite value: {key}")
        if not row["ci_low_db"] <= row["delta_psnr_db"] <= row["ci_high_db"]:
            raise ValueError("Effect must lie inside its reported interval.")
        if row["group"] not in {"matched", "intervention"}:
            raise ValueError("Unknown comparison group.")
    return rows


def render(rows):
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="566" '
        'viewBox="0 0 1440 566" role="img" aria-labelledby="title description">',
        '<title id="title">CorrPrior: reported PSNR effects and paired 95% confidence intervals</title>',
        '<desc id="description">Left: matched model comparisons on RealSR and SVSR. '
        'Right: feature interventions on one fixed CorrPrior checkpoint on RealSR. '
        'Positive values favor the first model or the original aligned features. '
        'The panels have different x-axis ranges. Interventions are not trained-baseline gains.</desc>',
        '<rect width="1440" height="566" fill="#ffffff"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#263238">',
    ]

    def text(x, y, value, size=18, weight="normal", anchor="start", color="#263238"):
        elements.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{color}">{escape(value)}</text>'
        )

    def line(x1, y1, x2, y2, color="#aab4bf", width=1.0, dashed=False):
        dash = ' stroke-dasharray="5 6"' if dashed else ""
        elements.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width}"{dash}/>'
        )

    panels = [
        ("matched", 18, 334, 678, -0.05, 0.20, [163, 294, 425],
         "(a) Matched model comparisons", "Positive favors the first model",
         "Same downstream SR training budget", [-0.05, 0, 0.05, 0.10, 0.15, 0.20]),
        ("intervention", 732, 1040, 1388, 0, 0.50, [215, 379],
         "(b) Fixed-checkpoint interventions", "Positive favors aligned CorrPrior features",
         "RealSR only; learned weights are unchanged", [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    ]
    for group, panel_x, x0, x1, lower, upper, ys, title, subtitle, note, ticks in panels:
        subset = [row for row in rows if row["group"] == group]
        if len(subset) != len(ys):
            raise ValueError(f"Expected {len(ys)} rows in {group}.")
        scale = lambda value: x0 + (value - lower) / (upper - lower) * (x1 - x0)
        elements.append(
            f'<rect x="{panel_x}" y="14" width="690" height="510" rx="14" '
            'fill="#f8fafc" stroke="#c5cfd8"/>'
        )
        text(panel_x + 24, 49, title, 23, "bold")
        text(panel_x + 24, 80, subtitle, 17, color="#52626e")
        text(panel_x + 24, 106, note, 16, color="#52626e")
        line(scale(0), 127, scale(0), 462, dashed=True)
        line(x0, 472, x1, 472)
        for tick in ticks:
            x = scale(tick)
            line(x, 472, x, 479)
            text(x, 500, f"{tick:.2f}" if group == "matched" else f"{tick:.1f}",
                 15, anchor="middle", color="#52626e")
        text((x0 + x1) / 2, 519, "PSNR difference (dB)", 15, anchor="middle")
        for row, y in zip(subset, ys):
            value, low, high = (row[key] for key in ("delta_psnr_db", "ci_low_db", "ci_high_db"))
            if not lower <= low <= high <= upper:
                raise ValueError("Interval exceeds the plotted axis; update the layout.")
            first, second = row["comparison"].split(" vs ", 1)
            text(panel_x + 24, y - 17, first, 19, "bold")
            text(panel_x + 24, y + 8, f"vs {second}", 19)
            text(panel_x + 24, y + 32, row["dataset"], 15, color="#52626e")
            color = "#007f5f" if low > 0 else "#657889"
            line(scale(low), y, scale(high), y, color, 3)
            for endpoint in (low, high):
                line(scale(endpoint), y - 9, scale(endpoint), y + 9, color, 2)
            elements.append(f'<circle cx="{scale(value):.2f}" cy="{y}" r="7" fill="{color}"/>')
            text(scale(value), y - 20, f"{value:+.4f}", 18, "bold", "middle", color)
    text(720, 550, "Points: reported effects. Bars: paired 95% CIs. Test-sample uncertainty, not variation across training seeds.",
         16, anchor="middle", color="#52626e")
    elements.extend(["</g>", "</svg>"])
    return "\n".join(elements) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    svg = render(read_effects())
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != svg:
            raise SystemExit("Plot is stale; run python scripts/render_result_figure.py")
        print("PASS: result figure matches the reported CSV values.")
    else:
        OUTPUT.write_text(svg, encoding="utf-8")
        print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
