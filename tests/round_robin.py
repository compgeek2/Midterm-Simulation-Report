"""A tiny router used only by the tests.

Person 2's real routers live in routers.py and are not imported here.
Round-robin ignores the cache and the queue: call 1 goes to GPU 0, call 2 to
GPU 1, then back to GPU 0, and so on. With a single GPU every call lands there,
which is what the hand-checked cases want.
"""

from __future__ import annotations


class RoundRobin:
    name = "round_robin"

    def __init__(self) -> None:
        self._next = 0

    def pick(self, call, gpus, now):
        gpu = gpus[self._next % len(gpus)]
        self._next += 1
        return gpu
