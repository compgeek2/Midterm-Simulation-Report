"""Speed-ratio sweep: P95 program completion time for every router in routers.py.

Two GPUs. GPU 0 is the fast one (speed 1). GPU 1 runs at 1/ratio, so a ratio
of 4 means the second GPU is four times slower. At every ratio the arrival
rate is 70% of the cluster's perfect-cache capacity:

    capacity (programs/s) = (speed_0 + speed_1) / mean_program_work()
                          = (1 + 1/ratio) / mean_program_work()
    arrival_rate          = 0.7 * capacity

mean_program_work() is the average fast-GPU seconds per program when every
prefix hits. A router that sends a call to a GPU that does not hold the
prefix does extra prefill, so the real load is higher than 0.7 for a router
that ignores locality. That is the comparison.

llm-d has a token-limit knob. An operator would tune that knob on a cluster
of identical GPUs and then leave it alone, so we try several limits at ratio
1 only, keep the limit with the lowest mean P95, and use that same limit at
every other ratio.

Person 2 owns routers.py. This file only imports it. A router is anything in
ROUTERS (instances or classes), or, if that list is absent, any class defined
in the module with a pick() method. The llm-d router is the one whose name or
class name contains "llm-d" / "llmd". Its constructor must accept
token_limit as a keyword.
"""

from __future__ import annotations

import copy
import csv
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import p95, simulate
from workload import make_workload, mean_program_work

HERE = Path(__file__).resolve().parent

RATIOS = [1.0, 1.5, 2.0, 3.0, 4.0]
LOAD = 0.7
N_PROGRAMS = 2_000
SEEDS = [0, 1, 2, 3, 4]
# Swept only at ratio 1. Context averages ~735 and the published Agentix
# cutoff is 2,048, so the grid brackets both.
LLMD_TOKEN_LIMITS = [256, 512, 1024, 2048, 4096, 8192]

# One color and one marker per router, in the order routers.py exports them.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]


def arrival_rate_for(ratio: float) -> float:
    """Programs per second that put the cluster at LOAD, with perfect caching."""
    combined_speed = 1.0 + 1.0 / ratio
    return LOAD * combined_speed / mean_program_work()


def speeds_for(ratio: float) -> List[float]:
    """GPU 0 is fast. GPU 1 takes `ratio` times as long to do the same work."""
    return [1.0, 1.0 / ratio]


def _folded(name: object) -> str:
    return str(name).lower().replace("_", "").replace("-", "")


def _is_llmd(spec: object) -> bool:
    """True if this router is llm-d, judged from its name or its class name."""
    candidates = []
    if inspect.isclass(spec):
        candidates.append(spec.__name__)
        candidates.append(getattr(spec, "name", ""))
    else:
        candidates.append(type(spec).__name__)
        candidates.append(getattr(spec, "name", ""))
    return any("llmd" in _folded(name) for name in candidates)


def load_router_specs() -> List[object]:
    """Return the router classes or prototype instances defined by Person 2."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    try:
        import routers as routers_mod
    except ImportError as exc:
        raise SystemExit(
            "Could not import routers.py. That file belongs to Person 2; "
            "this script runs the routers it defines and does not include any of its own."
        ) from exc

    if hasattr(routers_mod, "ROUTERS"):
        specs = list(routers_mod.ROUTERS)
    else:
        specs = []
        for name, obj in vars(routers_mod).items():
            if name.startswith("_"):
                continue
            if inspect.isclass(obj) and obj.__module__ == routers_mod.__name__ and hasattr(obj, "pick"):
                specs.append(obj)
    if not specs:
        raise SystemExit(
            "routers.py imported, but it has no ROUTERS list and no router classes. "
            "Export ROUTERS = [...] or define classes with a pick() method."
        )
    return specs


def instantiate(spec: object, token_limit: Optional[int] = None) -> object:
    """Build a fresh router so cursors and memory from the last run are gone.

    Classes are called. Instances are deep-copied from the prototype Person 2
    put in ROUTERS, which keeps whatever arguments they were constructed with.
    llm-d is always constructed with the token limit under test.
    """
    if _is_llmd(spec):
        if token_limit is None:
            raise ValueError("llm-d needs a token_limit")
        cls = spec if inspect.isclass(spec) else type(spec)
        try:
            return cls(token_limit=token_limit)
        except TypeError as exc:
            raise SystemExit(
                f"{cls.__name__} looks like the llm-d router but "
                f"{cls.__name__}(token_limit={token_limit}) failed. "
                "Its constructor needs a token_limit keyword."
            ) from exc

    if inspect.isclass(spec):
        return spec()
    return copy.deepcopy(spec)


def _router_name(router: object) -> str:
    return str(getattr(router, "name", type(router).__name__))


def _run_p95(spec: object, programs, speeds, token_limit: Optional[int]) -> float:
    router = instantiate(spec, token_limit)
    completions = simulate(programs, router, speeds)
    return p95(completions)


def _summarize(values: Sequence[float]) -> Dict[str, float]:
    return {
        "p95_mean": sum(values) / len(values),
        "p95_min": min(values),
        "p95_max": max(values),
    }


def tune_llmd(spec: object, workloads_at_ratio_1) -> int:
    """Pick the token limit with the best (lowest) mean P95 on identical GPUs.

    `workloads_at_ratio_1` is a list of program lists, one per seed. Every
    limit sees those same programs.
    """
    speeds = speeds_for(1.0)
    print(f"tuning {_router_name_from_spec(spec)} at speed ratio 1")
    best_limit = LLMD_TOKEN_LIMITS[0]
    best_mean = float("inf")
    for limit in LLMD_TOKEN_LIMITS:
        values = [
            _run_p95(spec, programs, speeds, limit) for programs in workloads_at_ratio_1
        ]
        summary = _summarize(values)
        print(
            f"  token_limit {limit:5d}   "
            f"mean P95 {summary['p95_mean']:.4f}   "
            f"range {summary['p95_min']:.4f}–{summary['p95_max']:.4f}"
        )
        if summary["p95_mean"] < best_mean:
            best_mean = summary["p95_mean"]
            best_limit = limit
    print(f"  chose token_limit {best_limit}")
    return best_limit


def _router_name_from_spec(spec: object) -> str:
    if not inspect.isclass(spec) and hasattr(spec, "name"):
        return str(spec.name)
    if inspect.isclass(spec) and isinstance(getattr(spec, "name", None), str):
        return str(spec.name)
    try:
        limit = LLMD_TOKEN_LIMITS[0] if _is_llmd(spec) else None
        return _router_name(instantiate(spec, limit))
    except SystemExit:
        raise
    except Exception:
        if inspect.isclass(spec):
            return spec.__name__
        return type(spec).__name__


def run() -> List[dict]:
    specs = load_router_specs()
    work = mean_program_work()
    print(f"mean fast-GPU seconds per program (perfect cache): {work:.4f}")

    # Same programs for every router at a given (ratio, seed).
    workloads = {}
    for ratio in RATIOS:
        rate = arrival_rate_for(ratio)
        print(f"ratio {ratio:g}: arrival_rate {rate:.4f} programs/s, speeds {speeds_for(ratio)}")
        for seed in SEEDS:
            workloads[(ratio, seed)] = make_workload(N_PROGRAMS, rate, seed)

    token_limits: Dict[int, int] = {}
    for index, spec in enumerate(specs):
        if _is_llmd(spec):
            token_limits[index] = tune_llmd(spec, [workloads[(1.0, seed)] for seed in SEEDS])

    rows: List[dict] = []
    for index, spec in enumerate(specs):
        limit = token_limits.get(index)
        # Name it the way it will actually run (with the chosen limit, for llm-d).
        name = _router_name(instantiate(spec, limit))
        for ratio in RATIOS:
            values = [
                _run_p95(spec, workloads[(ratio, seed)], speeds_for(ratio), limit)
                for seed in SEEDS
            ]
            summary = _summarize(values)
            print(
                f"{name:20s}  ratio {ratio:g}  "
                f"P95 mean {summary['p95_mean']:.4f}  "
                f"[{summary['p95_min']:.4f}, {summary['p95_max']:.4f}]"
            )
            rows.append(
                {
                    "router": name,
                    "speed_ratio": ratio,
                    "p95_mean": summary["p95_mean"],
                    "p95_min": summary["p95_min"],
                    "p95_max": summary["p95_max"],
                    "token_limit": "" if limit is None else limit,
                }
            )
    return rows


def write_csv(rows: Sequence[dict], path: Path) -> None:
    fieldnames = ["router", "speed_ratio", "p95_mean", "p95_min", "p95_max", "token_limit"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(rows: Sequence[dict], pdf_path: Path, png_path: Path) -> None:
    """ICML single-column figure: 3.25 inches wide, one line per router.

    The point is the mean P95 over the five seeds. The error bar runs from
    the worst seed to the best seed (min and max), not a standard deviation.
    """
    # Preserve first-seen order, which is the order routers were exported.
    names: List[str] = []
    for row in rows:
        if row["router"] not in names:
            names.append(row["router"])

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # 3.25 in is an ICML single column. Extra height is reserved for the legend,
    # which sits under the axes so it cannot cover a marker.
    fig, ax = plt.subplots(figsize=(3.25, 2.85))
    for index, name in enumerate(names):
        series = [row for row in rows if row["router"] == name]
        series.sort(key=lambda row: row["speed_ratio"])
        xs = [row["speed_ratio"] for row in series]
        ys = [row["p95_mean"] for row in series]
        lower = [row["p95_mean"] - row["p95_min"] for row in series]
        upper = [row["p95_max"] - row["p95_mean"] for row in series]
        ax.errorbar(
            xs,
            ys,
            yerr=[lower, upper],
            color=COLORS[index % len(COLORS)],
            marker=MARKERS[index % len(MARKERS)],
            markersize=4.5,
            linewidth=1.0,
            elinewidth=0.7,
            capsize=2,
            capthick=0.7,
            label=name,
        )

    ax.set_xlabel("Speed ratio")
    ax.set_ylabel("P95 program completion time (s)")
    ax.set_xticks(RATIOS)
    ax.set_xticklabels(["1", "1.5", "2", "3", "4"])
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        handlelength=1.6,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    # Fixed margins keep the saved figure at exactly 3.25 inches wide.
    # bbox_inches="tight" would grow or shrink that.
    fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.30)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def main() -> None:
    rows = run()
    csv_path = HERE / "results.csv"
    pdf_path = HERE / "figure.pdf"
    png_path = HERE / "figure.png"
    write_csv(rows, csv_path)
    plot_results(rows, pdf_path, png_path)
    print(f"wrote {csv_path.name}, {pdf_path.name}, {png_path.name}")


if __name__ == "__main__":
    main()
