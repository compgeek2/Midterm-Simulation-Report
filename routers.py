"""Routing policies (Person 2).

Every router only uses what interface.py exposes: the Call fields and the
GPUView methods. pick() returns one of the GPUs it was given.
"""
from typing import List

from interface import DECODE_RATE, PREFILL_RATE, Call, GPUView


class RoundRobin:
    """Send calls to GPU 0, 1, 2, ..., then wrap around. Ignores load and cache."""
    name = "round_robin"

    def __init__(self):
        self.next_index = 0                         # position in the GPU list for the next call

    def pick(self, call: Call, gpus: List[GPUView], now: float) -> GPUView:
        gpu = gpus[self.next_index % len(gpus)]     # % wraps back to 0 after the last GPU
        self.next_index += 1                        # move on for the next call
        return gpu


class LeastLoaded:
    """Send each call to the GPU with the fewest calls waiting + running."""
    name = "least_loaded"

    def pick(self, call: Call, gpus: List[GPUView], now: float) -> GPUView:
        # min() returns the first GPU with the smallest count, so ties go to the lowest list position
        return min(gpus, key=lambda gpu: gpu.outstanding_calls())


class Agentix:
    """Agentix (NSDI '26): short calls go anywhere, long calls stay pinned to the program's engine."""
    name = "agentix"
    SHORT_CALL_TOKENS = 2048                        # at or below this, the prompt is cheap to recompute

    def __init__(self):
        self.fallback = LeastLoaded()               # reuse LeastLoaded for "fewest outstanding calls"

    def pick(self, call: Call, gpus: List[GPUView], now: float) -> GPUView:
        # Short call, or first call of a program (no previous GPU): balance load
        if call.context_tokens <= self.SHORT_CALL_TOKENS or call.prev_gpu_id is None:
            return self.fallback.pick(call, gpus, now)
        # Long call: go back to the GPU that ran this program's previous call
        for gpu in gpus:
            if gpu.id == call.prev_gpu_id:
                return gpu
        return self.fallback.pick(call, gpus, now)  # safety net: previous GPU not in the list


class LLMD:
    """llm-d default, "sticky until saturated": stay where the cache is unless that GPU is too busy."""
    name = "llm_d"

    def __init__(self, token_limit: int):
        self.limit = token_limit                       # tokens_in_flight() at which a GPU counts as saturated

    def pick(self, call: Call, gpus: List[GPUView], now: float) -> GPUView:
        # The GPU holding the most of this program's cache
        best_cache = max(gpus, key=lambda gpu: gpu.cached(call.program_id))
        # Sticky: stay if it really has cache AND it is not saturated
        if best_cache.cached(call.program_id) > 0 and best_cache.tokens_in_flight() < self.limit:
            return best_cache
        # Otherwise: the GPU with the least queued work, measured in tokens
        return min(gpus, key=lambda gpu: gpu.tokens_in_flight())


class Ours:
    """Speed-aware estimated finish time: pick the GPU where this call should finish soonest."""
    name = "ours"

    def estimate(self, call: Call, gpu: GPUView, now: float) -> float:
        wait = gpu.est_seconds_left(now)                                  # time to drain the GPU's current queue
        hit = min(gpu.cached(call.program_id), call.context_tokens)       # cached tokens we can reuse (never more than the prompt)
        prefill = (call.context_tokens - hit) / PREFILL_RATE              # recompute only the uncached part
        decode = call.decode_guess / DECODE_RATE                          # an estimate; true decode length is never used
        return wait + (prefill + decode) / gpu.speed                      # slow GPU (speed < 1) takes longer

    def pick(self, call: Call, gpus: List[GPUView], now: float) -> GPUView:
        return min(gpus, key=lambda gpu: self.estimate(call, gpu, now))
