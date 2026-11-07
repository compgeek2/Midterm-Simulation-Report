"""BFCL-style tool-use programs, in the shape reported by Agentix (NSDI '26).

Agentix §6.1 / Fig. 11 measures real agent traces. For the Berkeley Function
Calling Leaderboard (BFCL) the paper's tool-use agents average 10.75 LLM calls,
and the longest program has 70. We don't have the paper's raw trace, so this
generator draws programs from simple distributions that hit those published
averages. One modeling choice is ours, and it matters:

    ASSUMPTION. Context grows along a program. Call 0 sends the system prompt
    and the user request. Every later call sends that whole history again, plus
    the answer the model just produced, plus the new tool result. So the
    "prompt length" of a call is not an independent draw — it is a running
    total. The distributions below are chosen so that, averaged over every
    call of every program, that running total is about 735 tokens, with a thin
    tail that reaches about 40,000.

What one program stores (see interface.Program):

    calls[i]       = (new_prompt_tokens, decode_tokens)
    tool_delays[i] = seconds the tool takes between call i and call i + 1

new_prompt_tokens is only the text added at that step (the original request,
or one tool result). It does not repeat the history. The engine rebuilds the
full prompt from these pieces.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Tuple

import numpy as np

from interface import DECODE_RATE, PREFILL_RATE, Program

# Published targets (Agentix Fig. 11 / §6.1) and the caps we impose.
TARGET_CALLS_MEAN = 10.75
TARGET_CONTEXT_MEAN = 735.0
TARGET_DECODE_MEAN = 34.0
MAX_CALLS = 70
MAX_CONTEXT_TOKENS = 40_000
TOOL_DELAY_MEAN = 1.0  # seconds, exponential

# How many calls a program makes.
#
# A geometric random variable (trials until first success) is the simplest
# long-tailed count: lots of short programs, then a thin tail of long ones.
# Numpy's geometric starts at 1, which is what we want (a program that never
# calls the model is not a program). Unclipped, the mean is 1/p. We refuse
# anything above 70, because that is the largest BFCL program in the paper,
# and that clip would otherwise shave a little off the mean. p is solved so
# the clipped mean is exactly 10.75:
#
#     E[min(K, 70)] = (1 - (1 - p) ** 70) / p = 10.75
#
# The clip only moves p from 1/10.75 ≈ 0.0930 to about 0.0929.
#
# Ordinary first prompt: system prompt + user request. Lognormal, so it is
# positive and mildly skewed. Mean 210 tokens, sigma 0.4. Most of these land
# between roughly 100 and 400 tokens.
FIRST_PROMPT_MEAN = 210.0
FIRST_PROMPT_SIGMA = 0.4

# The long tail of Fig. 11 (prompts up to ~40k) is rare, not typical. 1.5 in
# every 1,000 programs open on a long document instead of the modest prompt
# above. Their length is uniform from 20,000 to 40,000 tokens. Because every
# later call repeats that document, one of these programs lifts many calls —
# which is enough to reach the 40k tail, but rare enough that the average
# call stays near 735.
LONG_PROMPT_PROBABILITY = 0.0015
LONG_PROMPT_LOW = 20_000
LONG_PROMPT_HIGH = 40_000

# Tokens a tool returns between two calls. Also lognormal. Mean 16, sigma
# 0.55: a short function result, occasionally a few hundred tokens. The
# previous model answer (mean 34 tokens) is appended on top of this, so a
# typical step grows the prompt by about 50 tokens.
TOOL_TOKENS_MEAN = 16.0
TOOL_TOKENS_SIGMA = 0.55

# Decode length is deliberately short and NOT long-tailed. Poisson with mean
# 34 concentrates tightly around 34 (standard deviation about 6). We force a
# floor of 1 token; P(Poisson(34) = 0) is about 10^-15, so the floor does not
# move the mean.
DECODE_POISSON_MEAN = TARGET_DECODE_MEAN

# Draws used to turn the rounded lognormal / mixture into a numeric expectation
# for mean_program_work(). Large enough that the sampling noise is far below
# a token.
_EXPECTATION_DRAWS = 1_000_000
_EXPECTATION_SEED = 12345


def _geometric_success_probability() -> float:
    """p such that a geometric call count clipped at MAX_CALLS averages 10.75."""
    target = TARGET_CALLS_MEAN
    lo = 1e-6
    hi = 1.0 - 1e-9
    for _ in range(80):
        mid = (lo + hi) / 2.0
        # sum_{k=0}^{M-1} (1-p)^k = (1 - (1-p)^M) / p, and that sum is E[min(K, M)].
        clipped_mean = (1.0 - (1.0 - mid) ** MAX_CALLS) / mid
        if clipped_mean > target:
            # Mean is too high, so failures are too common: raise p.
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


_CALLS_P = _geometric_success_probability()


def _lognormal_mu(mean: float, sigma: float) -> float:
    """mu of the underlying normal, so the lognormal itself has expectation `mean`."""
    return math.log(mean) - 0.5 * sigma * sigma


def _sample_lognormal_tokens(rng: np.random.Generator, mean: float, sigma: float) -> int:
    """One positive integer token count, capped at the 40k prompt ceiling.

    `round` then clamp. The nominal `mean` is the mean before rounding and
    capping; both change it only slightly, and mean_program_work() measures
    the distribution actually drawn.
    """
    mu = _lognormal_mu(mean, sigma)
    rounded = int(round(float(rng.lognormal(mu, sigma))))
    if rounded < 1:
        return 1
    if rounded > MAX_CONTEXT_TOKENS:
        return MAX_CONTEXT_TOKENS
    return rounded


def _sample_first_prompt(rng: np.random.Generator) -> int:
    if float(rng.random()) < LONG_PROMPT_PROBABILITY:
        # integers() is exclusive of the high end, so +1 keeps 40_000 reachable.
        return int(rng.integers(LONG_PROMPT_LOW, LONG_PROMPT_HIGH + 1))
    return _sample_lognormal_tokens(rng, FIRST_PROMPT_MEAN, FIRST_PROMPT_SIGMA)


def _sample_tool_tokens(rng: np.random.Generator) -> int:
    return _sample_lognormal_tokens(rng, TOOL_TOKENS_MEAN, TOOL_TOKENS_SIGMA)


def _sample_decode_tokens(rng: np.random.Generator) -> int:
    drawn = int(rng.poisson(DECODE_POISSON_MEAN))
    return drawn if drawn >= 1 else 1


def call_records(program: Program) -> List[Tuple[int, int, int]]:
    """Expand stored calls into (context_tokens, new_tokens, decode_tokens).

    context_tokens is the full prompt the GPU would prefill, history included.
    new_tokens is how much that prompt grew since the previous call:

        call 0: new_tokens = first prompt
        call i: new_tokens = previous decode + this call's tool result

    The previous decode is part of the growth because the next prompt quotes
    the model's own answer. A perfect KV cache already holds those generated
    tokens, so the engine does not prefill them again when the prefix is still
    cached; `context_tokens - cached` is the prefill that is actually missing.
    """
    records: List[Tuple[int, int, int]] = []
    context = 0
    previous_decode = 0
    for index, (new_prompt, decode) in enumerate(program.calls):
        if index == 0:
            added = new_prompt
        else:
            added = previous_decode + new_prompt
        context += added
        records.append((context, added, decode))
        previous_decode = decode
    return records


def program_fast_seconds(program: Program) -> float:
    """Fast-GPU seconds to run this program with a perfect prefix cache.

    Perfect caching means every call after the first prefills only the new
    tool result. The history, including tokens the model itself just wrote, is
    already in the KV cache, so we do not pay PREFILL_RATE for it again.
    Decode is never cached away: every call still generates its answer.

    The `new_prompt` half of each stored pair is exactly that incremental
    prefill (first prompt, then each tool result). Summing those is the whole
    prefill bill.
    """
    prefill_tokens = 0
    decode_tokens = 0
    for new_prompt, decode in program.calls:
        prefill_tokens += new_prompt
        decode_tokens += decode
    return prefill_tokens / PREFILL_RATE + decode_tokens / DECODE_RATE


def _expected_token_draw(kind: str) -> float:
    """Numerical expectation of one sampler, via a fixed-seed Monte Carlo.

    Rounding makes the lognormal expectation annoying to write in closed form.
    A million draws from the same sampler the generator uses is stable to a
    fraction of a token, and it cannot drift from make_workload().
    """
    rng = np.random.default_rng(_EXPECTATION_SEED)
    if kind == "first":
        sample = _sample_first_prompt
    elif kind == "tool":
        sample = _sample_tool_tokens
    elif kind == "decode":
        sample = _sample_decode_tokens
    else:
        raise ValueError(kind)
    total = 0
    for _ in range(_EXPECTATION_DRAWS):
        total += sample(rng)
    return total / _EXPECTATION_DRAWS


@lru_cache(maxsize=1)
def mean_program_work() -> float:
    """Average fast-GPU seconds per program, assuming a perfect prefix cache.

    Experiments set the arrival rate as a fraction of cluster capacity, and
    capacity is "how many of these programs per second could the GPUs finish
    if every prefix hit." That number should not depend on one random sample
    of programs, so this is the expectation of program_fast_seconds under the
    distributions above.

    Call length is independent of the token sizes, so Wald's identity applies:
    the expected number of tool results in a program is E[K] - 1 = 9.75, and

        E[prefill tokens] = E[first prompt] + 9.75 * E[tool tokens]
        E[decode tokens]  = E[K] * E[decode length]

    Divide by the fast-GPU rates (speed 1) to get seconds. A slow GPU of speed
    s would take this long / s; the caller scales by speed separately.
    """
    first = _expected_token_draw("first")
    tool = _expected_token_draw("tool")
    decode = _expected_token_draw("decode")
    calls = TARGET_CALLS_MEAN
    prefill_tokens = first + (calls - 1.0) * tool
    decode_tokens = calls * decode
    return prefill_tokens / PREFILL_RATE + decode_tokens / DECODE_RATE


def make_workload(n_programs: int, arrival_rate: float, seed: int) -> List[Program]:
    """Draw `n_programs` programs and a Poisson arrival process.

    arrival_rate is programs per second. Inter-arrival gaps are exponential
    with that mean, which is the definition of a Poisson process. The first
    program does not arrive at time 0; it arrives after the first gap.

    The shape of a program (how many calls, how many tokens) and the clock
    (when it arrives) use different streams of the same seed. Changing the
    arrival rate therefore changes only the timestamps, not which programs
    show up. That keeps a speed-ratio sweep comparable: the same agents, a
    faster or slower arrival clock.
    """
    if n_programs < 0:
        raise ValueError("n_programs must be >= 0")
    if arrival_rate <= 0:
        raise ValueError("arrival_rate must be positive (programs per second)")
    if n_programs == 0:
        return []

    # default_rng accepts a sequence and mixes it into a SeedSequence, so
    # [seed, 1] and [seed, 2] are independent streams and both repeatable.
    shape_rng = np.random.default_rng([int(seed), 1])
    arrival_rng = np.random.default_rng([int(seed), 2])

    gaps = arrival_rng.exponential(1.0 / arrival_rate, size=n_programs)
    arrivals = np.cumsum(gaps)

    programs: List[Program] = []
    for index in range(n_programs):
        n_calls = int(shape_rng.geometric(_CALLS_P))
        if n_calls > MAX_CALLS:
            n_calls = MAX_CALLS

        first_prompt = _sample_first_prompt(shape_rng)
        calls: List[Tuple[int, int]] = []
        for call_index in range(n_calls):
            decode = _sample_decode_tokens(shape_rng)
            if call_index == 0:
                new_prompt = first_prompt
            else:
                new_prompt = _sample_tool_tokens(shape_rng)
            calls.append((new_prompt, decode))

        if n_calls == 1:
            delays: List[float] = []
        else:
            # One tool wait between each pair of consecutive calls. Mean 1 second.
            drawn = shape_rng.exponential(TOOL_DELAY_MEAN, size=n_calls - 1)
            delays = [float(value) for value in drawn]

        programs.append(
            Program(
                id=index,
                arrival_time=float(arrivals[index]),
                calls=calls,
                tool_delays=delays,
            )
        )
    return programs
