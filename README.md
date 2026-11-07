# Midterm simulation: routing agent programs across GPUs

A discrete-event simulator for BFCL-style tool-use agents (Agentix, NSDI '26, Fig. 11 / §6.1). Each program is a chain of LLM calls separated by tool waits. A router assigns every call to a GPU. GPUs differ in speed and keep a prefix KV cache.

`interface.py` is the shared contract. Do not change it without telling the other person.

| File | Owner | Role |
|---|---|---|
| `interface.py` | shared | `Program`, `Call`, `GPUView`, `Router`, and the rate constants |
| `workload.py` | Person 1 | Draw programs and Poisson arrivals |
| `engine.py` | Person 1 | Run the event loop and return completion times |
| `run_experiments.py` | Person 1 | Speed-ratio sweep, CSV, and figure |
| `tests/` | Person 1 | Workload checks and hand-checked engine cases |
| `routers.py` | Person 2 | The routers under test. Not in this commit |

Dependencies are the Python standard library, NumPy, and Matplotlib.

## Workload

`make_workload(n_programs, arrival_rate, seed)` returns a list of `Program`s.

Context grows along a program. Call 0 is the system prompt plus the user request. Each later call resends that whole history, plus the previous answer, plus the new tool result. Stored calls keep only the new piece, as `(new_prompt_tokens, decode_tokens)`. The engine rebuilds the full prompt.

The generator is tuned to the published BFCL averages:

- Calls per program: mean 10.75, geometric, clipped at 70
- Full prompt per call: mean about 735 tokens, with a thin tail up to about 40,000
- Decode length: mean 34, Poisson (short, not long-tailed)
- Tool delay between calls: exponential, mean 1 second
- Arrivals: Poisson at `arrival_rate` programs per second

The same seed yields the same programs at any arrival rate. Only the timestamps change.

`mean_program_work()` is the expected fast-GPU seconds per program when every prefix cache hits. Later calls prefill only the new tool text. Decode is always paid. Experiments set the arrival rate from this number.

## Simulator

`simulate(programs, router, speeds)` returns one completion time per program, in input order. Completion time is the finish of the last call minus the program's arrival. That includes queueing, prefill, decode, and tool waits. The tool wait does not hold a GPU.

A GPU is a FIFO queue. Speed 1 runs at `PREFILL_RATE` (10,000 tokens/s) and `DECODE_RATE` (1,500 tokens/s). A slower GPU has speed `1/ratio`, and runtime is divided by speed:

```text
(uncached prefill / PREFILL_RATE + decode / DECODE_RATE) / speed
```

The uncached prefill is `context_tokens` minus the cached prefix. The cache is read when the call starts, not when it is routed, because a call waiting in the queue can lose its prefix before it runs.

Each GPU holds `CACHE_TOKENS` (300,000) tokens. An entry is one program and a prefix length. When a new prefix does not fit, the least recently written program is evicted. After a call that is not the program's last, the GPU stores the full prompt plus the tokens just generated. When the program finishes, its entries are dropped on every GPU.

Routers never see the true decode length. `decode_guess` is 34 on the first call and the mean of that program's earlier decode lengths after that.

`p95(values)` is the linear 95th percentile.

## Tests

From this directory:

```bash
python -m unittest tests.test_workload tests.test_engine -v
```

`test_workload.py` draws 5,000 programs and checks the means against the targets above. One seed can sit a few tens of tokens off 735 because the 40k documents are rare. `test_engine.py` checks hand-computed cases: one program on one GPU, a cache hit versus a cold prompt, a slow GPU taking ratio-times longer, LRU eviction, the decode guess, and FIFO order. The round-robin router in `tests/round_robin.py` is for those tests only.

## Experiments

`run_experiments.py` imports every router from `routers.py`. Export them as `ROUTERS`, a list of instances or classes. Each router needs a `.name` and `pick(call, gpus, now)` returning one of the GPU objects it was given. A fresh instance is built for every run.

The llm-d router is the one whose name or class name contains `llm-d` / `llmd`. Its constructor must accept `token_limit` as a keyword. At speed ratio 1 the script tries 256, 512, 1024, 2048, 4096, and 8192, keeps the limit with the lowest mean P95, and uses that limit at every other ratio.

The sweep:

- Speed ratios 1, 1.5, 2, 3, and 4. GPU 0 has speed 1. GPU 1 has speed `1/ratio`.
- Arrival rate is 70% of perfect-cache capacity: `0.7 * (1 + 1/ratio) / mean_program_work()`. A router that misses the cache does extra prefill, so its real load is higher than 0.7.
- 2,000 programs per run, seeds 0 through 4. Every router sees the same programs at a given ratio and seed.

```bash
python run_experiments.py
```

This writes `results.csv` (mean, min, and max P95 over the five seeds), plus `figure.pdf` and `figure.png`. The figure is 3.25 inches wide, one ICML column. The x-axis is the speed ratio, the y-axis is P95 program completion time in seconds, and the error bar runs from the best seed to the worst.