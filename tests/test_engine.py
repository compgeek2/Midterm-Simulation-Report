"""Hand-checked simulations.

Every expected time below is computed from the rates in interface.py:

    runtime = (uncached_prefill / PREFILL_RATE + decode / DECODE_RATE) / speed

with PREFILL_RATE = 10,000 and DECODE_RATE = 1,500. A cached prefix shortens
uncached_prefill. It does not change the decode.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interface import (
    CACHE_TOKENS,
    DECODE_RATE,
    DEFAULT_DECODE_GUESS,
    PREFILL_RATE,
    Program,
)
from engine import p95, simulate
from tests.round_robin import RoundRobin


def _run(programs, speeds):
    return simulate(programs, RoundRobin(), speeds)


class EngineTests(unittest.TestCase):
    def test_one_program_one_gpu(self) -> None:
        # Arrives at t = 5 on a speed-1 GPU.
        # Call 0 prefills 1,000 and decodes 150, with an empty cache:
        #   1000/10000 + 150/1500 = 0.2 s, so it finishes at 5.2.
        # The GPU then holds the 1,150-token prefix (prompt + answer).
        # The tool takes 1 s, so call 1 is routed at 6.2.
        # Call 1's prompt is 1000 + 150 + 500 = 1,650. The prefix covers
        # 1,150 of those, leaving 500 to prefill, plus 150 to decode:
        #   500/10000 + 150/1500 = 0.15 s, finishing at 6.35.
        # Completion = 6.35 - 5 = 1.35 s.
        program = Program(
            id=0,
            arrival_time=5.0,
            calls=[(1000, 150), (500, 150)],
            tool_delays=[1.0],
        )
        got = _run([program], [1.0])
        self.assertAlmostEqual(got[0], 1.35, places=9)

    def test_cached_call_is_faster_than_a_cold_one(self) -> None:
        # Same second prompt as the test above (1,650 tokens, 150 decode).
        # Cold: the whole prompt is prefilled. 1650/10000 + 150/1500 = 0.265 s.
        # Warm: only the new 500 tokens are prefilled. 500/10000 + 0.1 = 0.15 s.
        cold = Program(id=0, arrival_time=0.0, calls=[(1650, 150)], tool_delays=[])
        warm = Program(
            id=0,
            arrival_time=0.0,
            calls=[(1000, 150), (500, 150)],
            tool_delays=[0.0],
        )
        cold_time = _run([cold], [1.0])[0]
        warm_total = _run([warm], [1.0])[0]
        # The warm program's first call costs 0.2 s and the tool wait is 0,
        # so whatever is left is the cached second call.
        cached_call = warm_total - (1000 / PREFILL_RATE + 150 / DECODE_RATE)
        self.assertAlmostEqual(cold_time, 0.265, places=9)
        self.assertAlmostEqual(cached_call, 0.15, places=9)
        self.assertLess(cached_call, cold_time)

    def test_slow_gpu_takes_ratio_times_longer(self) -> None:
        # 10,000 prefill + 1,500 decode is exactly 2 seconds on a fast GPU.
        # speed = 1/4, so the same call takes 2 * 4 = 8 seconds.
        ratio = 4.0
        program = Program(id=0, arrival_time=0.0, calls=[(10_000, 1_500)], tool_delays=[])
        fast = _run([program], [1.0])[0]
        slow = _run([program], [1.0 / ratio])[0]
        self.assertAlmostEqual(fast, 2.0, places=9)
        self.assertAlmostEqual(slow, fast * ratio, places=9)

    def test_eviction_and_cache_is_read_when_the_call_starts(self) -> None:
        # Capacity is 300,000 tokens. V runs a short call, leaves a 1,000-token
        # prefix, and comes back while a huge call is still occupying the GPU.
        #
        #   t = 0.00  V call 0 starts. 1,000/10,000 = 0.1 s. Cache = {V: 1000}.
        #   t = 0.12  F call 0 starts (V's tool wait is not over yet). It
        #             prefills 300,000 tokens, so it runs for 30 s.
        #   t = 0.15  V call 1 is routed. F is still running, so V waits.
        #             At this instant V's prefix is STILL cached. A router, or
        #             an engine that priced the call now, would expect a hit:
        #             only 1,000 of V's 2,000-token prompt is new.
        #   t = 30.12 F finishes and installs a 300,000-token prefix. V's 1,000
        #             tokens are the least recently used and do not fit, so V
        #             is evicted. V's waiting call then starts, sees no prefix,
        #             and prefills all 2,000 tokens (0.2 s). It finishes at 30.32.
        #
        # Completion of V is 30.32. It would be 30.22 if the prefix had survived
        # or if the runtime had been frozen back at t = 0.15.
        victim = Program(
            id=0,
            arrival_time=0.0,
            calls=[(1000, 0), (1000, 0)],
            tool_delays=[0.05],
        )
        filler = Program(
            id=1,
            arrival_time=0.12,
            calls=[(CACHE_TOKENS, 0), (1, 0)],
            tool_delays=[1_000.0],
        )
        self.assertEqual(CACHE_TOKENS, 300_000)

        v_first = 1000 / PREFILL_RATE
        f_runtime = CACHE_TOKENS / PREFILL_RATE
        f_finish = filler.arrival_time + f_runtime
        missed = (1000 + 1000) / PREFILL_RATE  # both prompts, nothing cached
        hit = 1000 / PREFILL_RATE              # only the new tool result
        evicted_completion = f_finish + missed
        cached_completion = f_finish + hit

        got = _run([victim, filler], [1.0])
        self.assertAlmostEqual(evicted_completion, 30.32, places=6)
        self.assertAlmostEqual(got[0], evicted_completion, places=9)
        self.assertGreater(got[0], cached_completion)
        # And the filler really did have to wait out its own 30 s prefill,
        # plus a second tiny call much later. We only check that it finished
        # after V, which is the timeline above.
        self.assertGreater(got[1], got[0])

    def test_routers_never_see_the_true_decode_length(self) -> None:
        # Decodes are 10, then 50, then 90. The guess is the mean of the ones
        # already finished, and 34 (DEFAULT_DECODE_GUESS) before any exist.
        #   call 0 guess = 34,          not 10
        #   call 1 guess = 10,          not 50
        #   call 2 guess = (10+50)/2 = 30, not 90
        # Context grows 100, then 100+10+40 = 150, then 150+50+70 = 270.
        seen = []

        class RecordingRouter:
            name = "recording"

            def pick(self, call, gpus, now):
                seen.append(call)
                return gpus[0]

        program = Program(
            id=7,
            arrival_time=0.0,
            calls=[(100, 10), (40, 50), (70, 90)],
            tool_delays=[0.0, 0.0],
        )
        simulate([program], RecordingRouter(), [1.0])

        self.assertEqual([c.index for c in seen], [0, 1, 2])
        self.assertEqual([c.context_tokens for c in seen], [100, 150, 270])
        self.assertEqual([c.new_tokens for c in seen], [100, 50, 120])
        self.assertEqual(
            [c.decode_guess for c in seen],
            [float(DEFAULT_DECODE_GUESS), 10.0, 30.0],
        )
        self.assertEqual([c.prev_gpu_id for c in seen], [None, 0, 0])
        for call in seen:
            self.assertFalse(hasattr(call, "decode_tokens"))
            self.assertNotIn(call.decode_guess, (50, 90))

    def test_fifo_queue_and_what_the_router_is_told(self) -> None:
        # Three programs, one GPU.
        #   P0 arrives at 0,    prefills 10,000 and decodes 1,500. Runs 0..2.
        #   P1 arrives at 0.5,  so it waits. True decode is 600, guess is 34.
        #   P2 arrives at 0.75, behind P0 (running) and P1 (queued).
        #
        # The view at P2's routing uses the GUESS for the queued work:
        #   remaining on P0 = 2 - 0.75 = 1.25
        #   P1 estimate     = 8000/10000 + 34/1500
        # and the true runtime, used only once P1 starts at t = 2, is
        #   8000/10000 + 600/1500 = 1.2, so P1 finishes at 3.2
        #   completion = 3.2 - 0.5 = 2.7
        # Using the guess as the runtime would finish P1 near t = 2.82 instead.
        notes = []

        class ProbingRouter:
            name = "probing"

            def pick(self, call, gpus, now):
                gpu = gpus[0]
                notes.append(
                    {
                        "id": call.program_id,
                        "now": now,
                        "outstanding": gpu.outstanding_calls(),
                        "left": gpu.est_seconds_left(now),
                        "cached_p0": gpu.cached(0),
                        "tokens": gpu.tokens_in_flight(),
                    }
                )
                return gpu

        programs = [
            Program(id=0, arrival_time=0.0, calls=[(10_000, 1_500)], tool_delays=[]),
            Program(id=1, arrival_time=0.5, calls=[(8_000, 600)], tool_delays=[]),
            Program(id=2, arrival_time=0.75, calls=[(1_000, 150)], tool_delays=[]),
        ]
        got = simulate(programs, ProbingRouter(), [1.0])

        # These three numbers are the FIFO schedule. P1 waits until t = 2,
        # then runs its true 1.2 s. P2 waits until t = 3.2, then runs 0.2 s.
        self.assertAlmostEqual(got[0], 2.0, places=9)
        self.assertAlmostEqual(got[1], 2.7, places=9)
        self.assertAlmostEqual(got[2], 3.4 - 0.75, places=9)

        by_id = {note["id"]: note for note in notes}
        self.assertEqual(by_id[0]["outstanding"], 0)
        self.assertEqual(by_id[1]["outstanding"], 1)
        self.assertAlmostEqual(by_id[1]["left"], 1.5, places=9)
        self.assertEqual(by_id[1]["cached_p0"], 0)
        self.assertEqual(by_id[1]["tokens"], 10_000 + DEFAULT_DECODE_GUESS)

        p1_estimate = 8_000 / PREFILL_RATE + DEFAULT_DECODE_GUESS / DECODE_RATE
        self.assertEqual(by_id[2]["outstanding"], 2)
        self.assertAlmostEqual(by_id[2]["left"], (2.0 - 0.75) + p1_estimate, places=9)
        self.assertEqual(
            by_id[2]["tokens"],
            (10_000 + DEFAULT_DECODE_GUESS) + (8_000 + DEFAULT_DECODE_GUESS),
        )

    def test_p95_interpolates(self) -> None:
        # 1..100. The linear 95th percentile sits 95% of the way from the
        # first sample to the last: index 99 * 0.95 = 94.05, between 95 and
        # 96, so 95.05.
        self.assertAlmostEqual(p95([float(i) for i in range(1, 101)]), 95.05, places=9)


if __name__ == "__main__":
    unittest.main()
