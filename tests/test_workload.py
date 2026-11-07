"""Check that 5,000 generated programs land near the Agentix BFCL targets.

The generator is random, so the match is statistical, not exact. Tolerances
are wide enough for the noise in one sample of this size and tight enough
that a wrong distribution (uniform calls, flat prompts, ...) fails.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interface import DECODE_RATE, PREFILL_RATE
from workload import (
    MAX_CALLS,
    TARGET_CALLS_MEAN,
    TARGET_CONTEXT_MEAN,
    TARGET_DECODE_MEAN,
    TOOL_DELAY_MEAN,
    make_workload,
    mean_program_work,
    program_fast_seconds,
)


def _full_contexts(program):
    """Full prompt length of each call, written out here so it doesn't just call the helper.

    Call 0 is the first prompt. Each later prompt appends the previous answer
    and the new tool result.
    """
    context = 0
    previous_decode = 0
    lengths = []
    for index, (new_prompt, decode) in enumerate(program.calls):
        if index == 0:
            context = new_prompt
        else:
            context = context + previous_decode + new_prompt
        lengths.append(context)
        previous_decode = decode
    return lengths


class WorkloadTests(unittest.TestCase):
    def test_means_match_bfcl_targets(self) -> None:
        n_programs = 5_000
        arrival_rate = 2.0  # programs per second; gaps should average 0.5 s
        programs = make_workload(n_programs, arrival_rate, seed=0)

        call_counts = []
        contexts = []
        decodes = []
        delays = []
        for program in programs:
            call_counts.append(len(program.calls))
            self.assertGreaterEqual(len(program.calls), 1)
            self.assertEqual(len(program.tool_delays), len(program.calls) - 1)
            contexts.extend(_full_contexts(program))
            for new_prompt, decode in program.calls:
                self.assertGreaterEqual(new_prompt, 0)
                self.assertGreaterEqual(decode, 1)
                decodes.append(decode)
            delays.extend(program.tool_delays)

        calls_mean = sum(call_counts) / len(call_counts)
        context_mean = sum(contexts) / len(contexts)
        decode_mean = sum(decodes) / len(decodes)
        delay_mean = sum(delays) / len(delays)
        gaps = [
            programs[i].arrival_time - programs[i - 1].arrival_time
            for i in range(1, len(programs))
        ]
        gap_mean = sum(gaps) / len(gaps)

        print()
        print(f"programs:                  {n_programs}", flush=True)
        print(f"mean calls per program:    {calls_mean:.3f}   (target {TARGET_CALLS_MEAN})", flush=True)
        print(f"mean context tokens/call:  {context_mean:.1f}   (target {TARGET_CONTEXT_MEAN})", flush=True)
        print(f"mean decode tokens/call:   {decode_mean:.3f}   (target {TARGET_DECODE_MEAN})", flush=True)
        print(f"max calls:                 {max(call_counts)}   (cap {MAX_CALLS})", flush=True)
        print(f"max context tokens:        {max(contexts)}", flush=True)
        print(f"mean tool delay (s):       {delay_mean:.3f}   (target {TOOL_DELAY_MEAN})", flush=True)
        print(f"mean interarrival (s):     {gap_mean:.3f}   (target {1.0 / arrival_rate})", flush=True)
        print(f"mean_program_work (s):     {mean_program_work():.4f}", flush=True)

        self.assertLessEqual(max(call_counts), MAX_CALLS)
        # Geometric tail: with 5,000 programs the longest is essentially always
        # near the cap, and the shortest is a single call.
        self.assertGreaterEqual(max(call_counts), 50)
        self.assertEqual(min(call_counts), 1)
        self.assertAlmostEqual(calls_mean, TARGET_CALLS_MEAN, delta=0.6)
        # Context noise is about ±25 tokens at this sample size, because a
        # handful of 20k–40k documents move the average. ±70 is within 10%.
        self.assertAlmostEqual(context_mean, TARGET_CONTEXT_MEAN, delta=70)
        self.assertAlmostEqual(decode_mean, TARGET_DECODE_MEAN, delta=1.0)
        # The long tail has to actually show up, not just the average.
        self.assertGreaterEqual(max(contexts), 20_000)
        self.assertAlmostEqual(delay_mean, TOOL_DELAY_MEAN, delta=0.05)
        self.assertAlmostEqual(gap_mean, 1.0 / arrival_rate, delta=0.05)
        self.assertLess(programs[0].arrival_time, programs[-1].arrival_time)

        # Perfect-caching work of this sample should sit next to the expectation
        # the experiments use to set the arrival rate.
        sample_work = sum(program_fast_seconds(p) for p in programs) / len(programs)
        self.assertAlmostEqual(sample_work, mean_program_work(), delta=0.02)

    def test_same_seed_same_shapes_at_any_arrival_rate(self) -> None:
        slow = make_workload(30, arrival_rate=1.0, seed=7)
        fast = make_workload(30, arrival_rate=5.0, seed=7)
        for left, right in zip(slow, fast):
            self.assertEqual(left.calls, right.calls)
            self.assertEqual(left.tool_delays, right.tool_delays)
        self.assertNotEqual(
            [p.arrival_time for p in slow],
            [p.arrival_time for p in fast],
        )

    def test_fast_seconds_counts_only_incremental_prefill(self) -> None:
        # 1,000 new prompt tokens and 1,500 decode tokens.
        # Prefill costs 1000/10000 = 0.1 s, decode costs 1500/1500 = 1 s.
        from interface import Program

        program = Program(
            id=0,
            arrival_time=0.0,
            calls=[(400, 600), (600, 900)],
            tool_delays=[1.0],
        )
        expected = (400 + 600) / PREFILL_RATE + (600 + 900) / DECODE_RATE
        self.assertAlmostEqual(program_fast_seconds(program), expected)
        self.assertAlmostEqual(expected, 0.1 + 1.0)


if __name__ == "__main__":
    unittest.main()
