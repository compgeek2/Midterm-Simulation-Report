"""Unit tests for routers.py, using hand-built fake GPUs (no simulator needed).

Run from the repo root:  python -m unittest -v tests.test_routers
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "import routers" works

from interface import DEFAULT_DECODE_GUESS, Call
from routers import LLMD, Agentix, LeastLoaded, Ours, RoundRobin

LLMD_LIMIT = 50_000   # tokens in flight at which llm-d calls a GPU saturated


class FakeGPU:
    """Implements GPUView with values set by hand."""

    def __init__(self, id, speed=1.0, cache=None, outstanding=0, in_flight=0, seconds_left=0.0):
        self.id = id
        self.speed = speed
        self._cache = cache or {}           # {program_id: cached tokens}
        self._outstanding = outstanding
        self._in_flight = in_flight
        self._seconds_left = seconds_left

    def cached(self, program_id): return self._cache.get(program_id, 0)
    def outstanding_calls(self): return self._outstanding
    def tokens_in_flight(self): return self._in_flight
    def est_seconds_left(self, now): return self._seconds_left


def make_call(context, new, prev_gpu_id, program_id=1, index=1):
    return Call(program_id=program_id, index=index, context_tokens=context, new_tokens=new,
                decode_guess=DEFAULT_DECODE_GUESS, prev_gpu_id=prev_gpu_id)


def failure_case():
    """Program 1 has 16k tokens cached on a slow, busy GPU (but under the llm-d limit);
    a fast GPU is idle with nothing cached. The next call adds 500 new tokens."""
    slow = FakeGPU(0, speed=0.33, cache={1: 16_000}, outstanding=6, in_flight=40_000, seconds_left=8.0)
    fast = FakeGPU(1, speed=1.0)
    return make_call(16_500, 500, prev_gpu_id=0), slow, fast


def opposite_case():
    """Program 1's cache is on a FAST GPU that is only slightly busy; another fast GPU is idle."""
    cached = FakeGPU(0, speed=1.0, cache={1: 16_000}, outstanding=1, in_flight=2_000, seconds_left=0.3)
    idle = FakeGPU(1, speed=1.0)
    return make_call(16_500, 500, prev_gpu_id=0), cached, idle


class TestRoundRobin(unittest.TestCase):
    def test_alternates_between_two_gpus(self):
        r, gpus = RoundRobin(), [FakeGPU(0), FakeGPU(1)]
        picks = [r.pick(make_call(100, 100, None), gpus, 0).id for _ in range(4)]
        self.assertEqual(picks, [0, 1, 0, 1])

    def test_ignores_load(self):
        r = RoundRobin()
        gpus = [FakeGPU(0, outstanding=99), FakeGPU(1)]
        self.assertEqual(r.pick(make_call(100, 100, None), gpus, 0).id, 0)  # busy GPU still gets its turn


class TestLeastLoaded(unittest.TestCase):
    def test_picks_fewest_outstanding(self):
        gpus = [FakeGPU(0, outstanding=5), FakeGPU(1, outstanding=2), FakeGPU(2, outstanding=7)]
        self.assertEqual(LeastLoaded().pick(make_call(100, 100, None), gpus, 0).id, 1)

    def test_ignores_cache(self):
        gpus = [FakeGPU(0, cache={1: 16_000}, outstanding=3), FakeGPU(1, outstanding=1)]
        self.assertEqual(LeastLoaded().pick(make_call(16_500, 500, 0), gpus, 0).id, 1)

    def test_tie_goes_to_first(self):
        gpus = [FakeGPU(0, outstanding=2), FakeGPU(1, outstanding=2)]
        self.assertEqual(LeastLoaded().pick(make_call(100, 100, None), gpus, 0).id, 0)


class TestAgentix(unittest.TestCase):
    def test_short_call_goes_least_loaded(self):
        gpus = [FakeGPU(0, outstanding=4), FakeGPU(1, outstanding=1)]
        call = make_call(2048, 100, prev_gpu_id=0)          # exactly at the threshold counts as short
        self.assertEqual(Agentix().pick(call, gpus, 0).id, 1)

    def test_long_call_stays_on_previous_gpu(self):
        gpus = [FakeGPU(0, outstanding=4), FakeGPU(1, outstanding=1)]
        call = make_call(2049, 100, prev_gpu_id=0)
        self.assertEqual(Agentix().pick(call, gpus, 0).id, 0)

    def test_first_call_uses_least_loaded(self):
        gpus = [FakeGPU(0, outstanding=4), FakeGPU(1, outstanding=1)]
        call = make_call(10_000, 10_000, prev_gpu_id=None, index=0)
        self.assertEqual(Agentix().pick(call, gpus, 0).id, 1)

    def test_failure_case_stays_on_slow_gpu(self):
        call, slow, fast = failure_case()
        self.assertEqual(Agentix().pick(call, [slow, fast], 0).id, slow.id)


class TestLLMD(unittest.TestCase):
    def test_stays_on_cache_when_under_limit(self):
        gpus = [FakeGPU(0, cache={1: 8_000}, in_flight=10_000), FakeGPU(1, in_flight=0)]
        self.assertEqual(LLMD(LLMD_LIMIT).pick(make_call(8_500, 500, 0), gpus, 0).id, 0)

    def test_leaves_when_saturated(self):
        gpus = [FakeGPU(0, cache={1: 8_000}, in_flight=LLMD_LIMIT), FakeGPU(1, in_flight=5_000)]
        self.assertEqual(LLMD(LLMD_LIMIT).pick(make_call(8_500, 500, 0), gpus, 0).id, 1)

    def test_no_cache_anywhere_picks_fewest_tokens(self):
        gpus = [FakeGPU(0, in_flight=9_000), FakeGPU(1, in_flight=3_000)]
        self.assertEqual(LLMD(LLMD_LIMIT).pick(make_call(500, 500, None), gpus, 0).id, 1)

    def test_failure_case_stays_on_slow_gpu(self):
        call, slow, fast = failure_case()
        self.assertLess(slow.tokens_in_flight(), LLMD_LIMIT)  # sanity: really under the limit
        self.assertEqual(LLMD(LLMD_LIMIT).pick(call, [slow, fast], 0).id, slow.id)


class TestOurs(unittest.TestCase):
    def test_failure_case_moves_to_fast_gpu(self):
        call, slow, fast = failure_case()
        ours = Ours()
        # slow: 8 + (500/10000 + 34/1500)/0.33 ~= 8.22 s ; fast: 0 + 16500/10000 + 34/1500 ~= 1.67 s
        self.assertAlmostEqual(ours.estimate(call, slow, 0), 8.22, places=2)
        self.assertAlmostEqual(ours.estimate(call, fast, 0), 1.67, places=2)
        self.assertEqual(ours.pick(call, [slow, fast], 0).id, fast.id)

    def test_opposite_case_stays_on_cached_fast_gpu(self):
        call, cached, idle = opposite_case()
        ours = Ours()
        # cached: 0.3 + 500/10000 + 34/1500 ~= 0.37 s ; idle: 16500/10000 + 34/1500 ~= 1.67 s
        self.assertLess(ours.estimate(call, cached, 0), ours.estimate(call, idle, 0))
        self.assertEqual(ours.pick(call, [cached, idle], 0).id, cached.id)

    def test_cache_larger_than_context_is_capped(self):
        gpu = FakeGPU(0, cache={1: 50_000})               # stale/extra cache must not make prefill negative
        call = make_call(1_000, 100, 0)
        self.assertAlmostEqual(Ours().estimate(call, gpu, 0), DEFAULT_DECODE_GUESS / 1_500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
