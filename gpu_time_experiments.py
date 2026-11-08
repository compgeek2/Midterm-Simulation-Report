"""P95 GPU time for every router: completion time minus tool delays.

A program's completion time is (finish of its last call) - (its arrival).
That includes the tool waits between calls. Those waits are drawn by the
workload and the engine adds them unchanged (call i+1 becomes ready exactly
tool_delays[i] seconds after call i finishes), so they are identical under
every router. Subtracting them leaves

    gpu_time = queueing + prefill + decode, summed over the program's calls

which is the part routing can change.

Everything about the experiment comes from run_experiments.py: the speed
ratios, the seeds, the 70% load, the workload size and the llm_d token-limit
tuning on equal GPUs. Nothing is copied here; it is all imported.

Writes gpu_time_results.csv with one row per (router, speed_ratio, seed).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from engine import p95, simulate
from run_experiments import (
    N_PROGRAMS,
    RATIOS,
    SEEDS,
    _is_llmd,
    _router_name,
    arrival_rate_for,
    instantiate,
    load_router_specs,
    speeds_for,
    tune_llmd,
)
from workload import make_workload

HERE = Path(__file__).resolve().parent
FIELDS = ["router", "speed_ratio", "seed", "p95_gpu_time", "p95_completion"]


def gpu_times(programs, completions: List[float]) -> List[float]:
    """Each program's completion time minus the sum of its tool delays.

    Stops the run if any value is negative, since that would mean the
    subtraction is removing time the program never actually waited.
    """
    result = []
    for program, completion in zip(programs, completions):
        gpu_time = completion - sum(program.tool_delays)
        if gpu_time < 0:
            raise SystemExit(
                f"negative GPU time {gpu_time:.6f} s for program {program.id}: "
                f"completion {completion:.6f} s, tool delays sum "
                f"{sum(program.tool_delays):.6f} s"
            )
        result.append(gpu_time)
    return result


def run() -> List[dict]:
    specs = load_router_specs()

    # Same programs for every router at a given (ratio, seed), exactly as in
    # run_experiments.run().
    workloads: Dict[Tuple[float, int], list] = {}
    for ratio in RATIOS:
        for seed in SEEDS:
            workloads[(ratio, seed)] = make_workload(N_PROGRAMS, arrival_rate_for(ratio), seed)

    # llm_d's token limit is tuned on equal GPUs (ratio 1) by P95 completion
    # time, the same way run_experiments.py does it, then held fixed.
    token_limits: Dict[int, int] = {}
    for index, spec in enumerate(specs):
        if _is_llmd(spec):
            token_limits[index] = tune_llmd(spec, [workloads[(1.0, seed)] for seed in SEEDS])

    rows: List[dict] = []
    for index, spec in enumerate(specs):
        limit = token_limits.get(index)
        name = _router_name(instantiate(spec, limit))
        for ratio in RATIOS:
            for seed in SEEDS:
                programs = workloads[(ratio, seed)]
                # A fresh router per run, so no state leaks between runs.
                completions = simulate(programs, instantiate(spec, limit), speeds_for(ratio))
                row = {
                    "router": name,
                    "speed_ratio": ratio,
                    "seed": seed,
                    "p95_gpu_time": p95(gpu_times(programs, completions)),
                    "p95_completion": p95(completions),
                }
                rows.append(row)
                print(
                    f"{name:14s} ratio {ratio:<4g} seed {seed}  "
                    f"P95 GPU time {row['p95_gpu_time']:9.4f} s   "
                    f"P95 completion {row['p95_completion']:9.4f} s"
                )
    return rows


def main() -> None:
    rows = run()
    path = HERE / "gpu_time_results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"no negative GPU times; wrote {path.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
