"""Plot P95 GPU time vs speed ratio from gpu_time_results.csv.

GPU time = completion time minus tool delays (queueing + prefill + decode).
Each line is the mean over seeds of the per-seed P95. The per-seed values
are in the CSV and the printed table; the plot shows only the means.

round_robin is left out of the plot: its values are tens to thousands of
times larger and would flatten everyone else. It is still in the printed
table. "ours" is drawn thick in a strong color; the other routers are thinner
and slightly faded so ours stands out.

Also prints, for each ratio, every router's mean P95 GPU time (round_robin
included) and how many seeds Ours beats llm_d on.

Writes figure_gpu_time.pdf and figure_gpu_time.png.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same colors and markers as figure.png, so a router looks the same everywhere.
from run_experiments import COLORS, MARKERS

HERE = Path(__file__).resolve().parent
LEFT_OUT = "round_robin"  # in the table, not in the plot
OURS = "ours"
BASELINE = "llm_d"
# Ours keeps its pink from figure.png, but a deeper, stronger shade of it.
OURS_COLOR = "#d6336c"
OURS_WIDTH = 3.0
OTHER_WIDTH = 1.5
OTHER_ALPHA = 0.75  # how opaque the other routers' lines are
# Names shown in the legend. The CSV keeps the routers' own names.
DISPLAY_NAMES = {
    "ours": "EFT",
    "llm_d": "llm-d",
    "agentix": "Agentix",
    "least_loaded": "Least loaded",
    "round_robin": "Round robin",
}


def read_rows(path: Path) -> List[dict]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["speed_ratio"] = float(row["speed_ratio"])
            row["seed"] = int(row["seed"])
            row["p95_gpu_time"] = float(row["p95_gpu_time"])
            row["p95_completion"] = float(row["p95_completion"])
            rows.append(row)
    return rows


def per_seed(rows: List[dict]) -> Dict[Tuple[str, float], Dict[int, float]]:
    """(router, ratio) -> {seed: P95 GPU time}."""
    table: Dict[Tuple[str, float], Dict[int, float]] = defaultdict(dict)
    for row in rows:
        table[(row["router"], row["speed_ratio"])][row["seed"]] = row["p95_gpu_time"]
    return table


def in_order(values) -> List:
    """Unique values in first-seen order."""
    seen: List = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def print_table(names: List[str], ratios: List[float], table) -> None:
    """Mean P95 GPU time per router at each ratio, plus Ours-vs-llm_d seed wins."""
    width = max(len(name) for name in names) + 2
    header = "ratio " + "".join(f"{name:>{width}s}" for name in names)
    if OURS in names and BASELINE in names:
        header += f"   {OURS} beats {BASELINE}"
    print("Mean P95 GPU time (s) over seeds")
    print(header)
    for ratio in ratios:
        line = f"{f'{ratio:g}x':<6s}"
        for name in names:
            values = table[(name, ratio)].values()
            line += f"{sum(values) / len(values):>{width}.3f}"
        if OURS in names and BASELINE in names:
            ours = table[(OURS, ratio)]
            base = table[(BASELINE, ratio)]
            seeds = sorted(set(ours) & set(base))
            wins = sum(1 for seed in seeds if ours[seed] < base[seed])
            line += f"   {wins} of {len(seeds)} seeds"
        print(line)


def plot(names: List[str], ratios: List[float], table, pdf_path: Path, png_path: Path) -> None:
    # Color and marker come from each router's position in the full list
    # (round_robin included), so they match figure.png.
    style = {
        name: (COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)])
        for i, name in enumerate(names)
    }
    if OURS in style:
        style[OURS] = (OURS_COLOR, style[OURS][1])

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # 3.25 in is an ICML single column.
    fig, ax = plt.subplots(figsize=(3.25, 2.45))

    for name in names:
        if name == LEFT_OUT:
            continue
        values = [list(table[(name, r)].values()) for r in ratios]
        means = [sum(v) / len(v) for v in values]
        color, marker = style[name]
        is_ours = name == OURS
        # Ours is drawn on top of the other lines.
        ax.plot(
            ratios, means,
            color=color,
            marker=marker,
            markersize=5 if is_ours else 3.5,
            linewidth=OURS_WIDTH if is_ours else OTHER_WIDTH,
            alpha=1.0 if is_ours else OTHER_ALPHA,
            zorder=4 if is_ours else 3,
            label=DISPLAY_NAMES.get(name, name),
        )

    ax.set_xlabel("How many times slower the slow GPU is")
    ax.set_ylabel("P95 GPU time (s)")
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{ratio:g}x" for ratio in ratios])
    ax.set_xlim(ratios[0] - 0.15, ratios[-1] + 0.15)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # One-row legend under the axes, so it covers no data and is not clipped.
    # Centered on the whole figure (not the axes, which sit right of center
    # because of the y label) so the longer display names fit inside it.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(labels),
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=0.8,
        borderaxespad=0.0,
    )

    # Fixed margins keep the saved figure exactly 3.25 in wide.
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.28)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def main() -> None:
    rows = read_rows(HERE / "gpu_time_results.csv")
    names = in_order(row["router"] for row in rows)
    ratios = sorted(in_order(row["speed_ratio"] for row in rows))
    table = per_seed(rows)
    print_table(names, ratios, table)
    pdf_path = HERE / "figure_gpu_time.pdf"
    png_path = HERE / "figure_gpu_time.png"
    plot(names, ratios, table, pdf_path, png_path)
    print(f"wrote {pdf_path.name}, {png_path.name}")


if __name__ == "__main__":
    main()
