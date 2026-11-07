"""Discrete-event simulator for routing a program's LLM calls across GPUs.

A program is a chain, not a bag of independent requests. It arrives, runs
call 0 on whichever GPU the router picks, waits while the tool runs (that
wait does not hold a GPU), then comes back for call 1, and so on. The number
we report is the program's completion time: when its last call finishes,
minus when the program arrived.

Time moves only when something happens. There are two events:

    ready   a call needs a GPU (the program just arrived, or a tool returned)
    finish  the call running on a GPU has ended

If both happen at the same instant, finishes go first. The GPU is free before
any new routing decision at that timestamp, and if the program that just
finished still has another call coming, its new prefix is already installed.

The router is not allowed to see the true decode length. Each Call carries
decode_guess, the average of this program's earlier decode lengths (or
DEFAULT_DECODE_GUESS on the first call). The real length stays in the
workload and is used only here, when the call actually runs.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict, deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from interface import (
    CACHE_TOKENS,
    DECODE_RATE,
    DEFAULT_DECODE_GUESS,
    PREFILL_RATE,
    Call,
    Program,
)
from workload import call_records

# Heap order at one timestamp: a finish happens before a new call is routed.
_FINISH = 0
_READY = 1


class GPU:
    """One GPU: a FIFO queue, a speed, and an LRU prefix cache.

    This is the object routers see (it satisfies GPUView in interface.py).
    Anything prefixed with _ is engine state. Routers must not need it, and
    peeking at the cache does not change eviction order — only finishing a
    call does, because that is when new KV entries are written.

    Speed is a multiplier on the fast GPU. speed 1 runs at PREFILL_RATE and
    DECODE_RATE. speed 1/4 takes four times as long. The queue is FIFO and
    holds one running call plus any calls waiting behind it. There is no
    preemption: a call that has started runs to completion.

    Cache. Each entry is one program, and the value is how many tokens of
    that program's prompt are stored, counting from the beginning. A prefix
    cache is useless if it has a hole in the middle, so we never store
    anything but a prefix. Capacity is CACHE_TOKENS, shared by every program
    on this GPU. When a new prefix does not fit, the program that was written
    least recently is dropped, and we repeat until the new one fits. A single
    program cannot hold more than the whole GPU; if its context is larger
    than CACHE_TOKENS we keep the leading CACHE_TOKENS tokens.
    """

    def __init__(self, gpu_id: int, speed: float) -> None:
        if speed <= 0:
            raise ValueError("GPU speed must be positive")
        self.id = gpu_id
        self.speed = float(speed)
        # program_id -> number of prefix tokens stored
        self._cache: Dict[int, int] = {}
        # Front is least recently written, back is most recently written.
        self._lru: "OrderedDict[int, None]" = OrderedDict()
        self._tokens_used = 0
        self._waiting: Deque[_Job] = deque()
        self._running: Optional[_Job] = None

    def __repr__(self) -> str:
        return f"GPU(id={self.id}, speed={self.speed})"

    # --- what a router is allowed to ask ---

    def cached(self, program_id: int) -> int:
        """Tokens of this program's prompt currently stored here (a prefix)."""
        return self._cache.get(program_id, 0)

    def outstanding_calls(self) -> int:
        """Calls waiting in the FIFO queue, plus the one running, if any."""
        running = 1 if self._running is not None else 0
        return running + len(self._waiting)

    def tokens_in_flight(self) -> int:
        """Estimated uncached-prefill plus decode tokens for queued and running calls.

        The estimate uses each call's decode_guess, never the true length, and
        the cache as it stands right now. It counts the whole call, including
        a call that has already started; est_seconds_left is the one that
        subtracts time already elapsed.
        """
        total = 0.0
        for job in self._outstanding_jobs():
            total += self._estimated_tokens(job.call)
        return int(round(total))

    def est_seconds_left(self, now: float) -> float:
        """Estimated seconds until this GPU goes idle, measured from `now`.

        The running call's finish time was fixed when it started (with the
        true decode length), so the remaining time on it is exact. Calls still
        waiting are estimated from the current cache and from decode_guess.
        A waiting call's real cost can change before it starts, because a
        finish in front of it may evict the prefix it was hoping to reuse.
        Routers are looking at an estimate, and the engine recomputes the real
        runtime when the call actually starts.
        """
        remaining = 0.0
        if self._running is not None:
            remaining += max(0.0, self._running.finish_time - now)
        for job in self._waiting:
            remaining += self._estimated_seconds(job.call)
        return remaining

    # --- engine internals ---

    def _outstanding_jobs(self) -> List["_Job"]:
        jobs: List[_Job] = []
        if self._running is not None:
            jobs.append(self._running)
        jobs.extend(self._waiting)
        return jobs

    def _estimated_tokens(self, call: Call) -> float:
        uncached = max(0, call.context_tokens - self.cached(call.program_id))
        return uncached + call.decode_guess

    def _estimated_seconds(self, call: Call) -> float:
        uncached = max(0, call.context_tokens - self.cached(call.program_id))
        return (uncached / PREFILL_RATE + call.decode_guess / DECODE_RATE) / self.speed

    def _enqueue(self, job: "_Job", now: float) -> Optional[float]:
        """Append a call to the FIFO queue. Start it if the GPU is idle.

        Returns the finish time when the call starts immediately, or None when
        it has to wait. The caller schedules the finish event.
        """
        if self._running is None:
            self._begin(job, now)
            return job.finish_time
        self._waiting.append(job)
        return None

    def _begin(self, job: "_Job", now: float) -> None:
        """Start `job` at time `now`, reading the cache at this moment.

        Runtime = (uncached prefill / PREFILL_RATE + decode / DECODE_RATE) / speed.
        Uncached prefill is the part of the prompt past the stored prefix.
        This is deliberately computed here and not at routing time: the prefix
        can be evicted by whatever runs in between.
        """
        cached = min(self.cached(job.call.program_id), job.call.context_tokens)
        uncached = job.call.context_tokens - cached
        runtime = (uncached / PREFILL_RATE + job.decode_tokens / DECODE_RATE) / self.speed
        job.finish_time = now + runtime
        self._running = job

    def _take_finished(self, now: float) -> "_Job":
        """The running call ended. Pull it off the GPU, but do not start the next one yet.

        The event loop installs the new prefix before the next queued call
        starts. Starting here would price that next call against a cache that
        does not yet include the call that just finished, and that might still
        contain a prefix the new one is about to evict.
        """
        if self._running is None:
            raise RuntimeError(f"GPU {self.id} got a finish event while idle")
        finished = self._running
        self._running = None
        finished.finish_time = now
        return finished

    def _start_next(self, now: float) -> Optional[float]:
        """Start the head of the FIFO queue if the GPU is idle. Return its finish time."""
        if self._running is not None or not self._waiting:
            return None
        job = self._waiting.popleft()
        self._begin(job, now)
        return job.finish_time

    def _set_cache(self, program_id: int, tokens: int) -> None:
        """Store a prefix of `tokens` for this program, evicting LRU entries to fit.

        Replaces any prefix this program already had on this GPU. The program
        is then the most recently used. Passing 0 drops the entry.
        """
        self._drop(program_id)
        amount = int(tokens)
        if amount <= 0:
            return
        if amount > CACHE_TOKENS:
            # Keep the leading prefix that physically fits on this GPU.
            amount = CACHE_TOKENS
        while self._tokens_used + amount > CACHE_TOKENS and self._lru:
            victim, _ = self._lru.popitem(last=False)
            self._tokens_used -= self._cache.pop(victim)
        self._cache[program_id] = amount
        self._tokens_used += amount
        self._lru[program_id] = None
        self._check_cache()

    def _drop(self, program_id: int) -> None:
        """Forget this program's prefix, if we had one. Used when it finishes."""
        if program_id not in self._cache:
            return
        self._tokens_used -= self._cache.pop(program_id)
        del self._lru[program_id]
        self._check_cache()

    def _check_cache(self) -> None:
        """The token counter, the map, and the LRU order describe one cache."""
        if self._tokens_used != sum(self._cache.values()):
            raise RuntimeError("KV cache token counter drifted from the stored prefixes")
        if set(self._cache) != set(self._lru):
            raise RuntimeError("KV cache and LRU order list different programs")


class _Prepared:
    """Everything the engine knows about one call, including the hidden decode length."""

    __slots__ = ("context_tokens", "new_tokens", "decode_tokens", "decode_guess")

    def __init__(
        self,
        context_tokens: int,
        new_tokens: int,
        decode_tokens: int,
        decode_guess: float,
    ) -> None:
        self.context_tokens = context_tokens
        self.new_tokens = new_tokens
        self.decode_tokens = decode_tokens
        self.decode_guess = decode_guess


class _Job:
    """A call sitting on a GPU. `decode_tokens` is never copied onto the Call."""

    __slots__ = ("program_index", "call_index", "call", "decode_tokens", "finish_time")

    def __init__(
        self,
        program_index: int,
        call_index: int,
        call: Call,
        decode_tokens: int,
    ) -> None:
        self.program_index = program_index
        self.call_index = call_index
        self.call = call
        self.decode_tokens = decode_tokens
        self.finish_time = 0.0


def _prepare(program: Program) -> List[_Prepared]:
    """Build the per-call table, including the decode guess the router will see.

    Guess for call 0 is DEFAULT_DECODE_GUESS (there is no history yet).
    Guess for call i is the mean of decode lengths 0 .. i-1. The true length
    of call i is not part of that mean — that is the whole point of the guess.
    """
    records = call_records(program)
    prepared: List[_Prepared] = []
    decode_sum = 0
    for index, (context, new_tokens, decode) in enumerate(records):
        if index == 0:
            guess = float(DEFAULT_DECODE_GUESS)
        else:
            guess = decode_sum / index
        prepared.append(_Prepared(context, new_tokens, decode, guess))
        decode_sum += decode
    return prepared


def _make_call(program: Program, call_index: int, prepared: _Prepared, prev_gpu_id: Optional[int]) -> Call:
    return Call(
        program_id=program.id,
        index=call_index,
        context_tokens=prepared.context_tokens,
        new_tokens=prepared.new_tokens,
        decode_guess=prepared.decode_guess,
        prev_gpu_id=prev_gpu_id,
    )


def simulate(programs: List[Program], router: object, speeds: List[float]) -> List[float]:
    """Run `programs` to completion. Return each program's completion time, in input order.

    completion time = (finish time of the last call) - (the program's arrival).
    It includes queueing, prefill, decode, and every tool wait in between.
    It does not include anything after the last token is produced.

    `router` must provide name and pick(call, gpus, now) -> one of `gpus`,
    as in interface.Router. `speeds[i]` is the speed of GPU i.
    """
    if not speeds:
        raise ValueError("need at least one GPU speed")
    seen = set()
    for program in programs:
        if program.id in seen:
            raise ValueError(f"duplicate program id {program.id}")
        seen.add(program.id)
        if not program.calls:
            raise ValueError(f"program {program.id} has no calls")
        if len(program.tool_delays) != len(program.calls) - 1:
            raise ValueError(
                f"program {program.id} has {len(program.calls)} calls but "
                f"{len(program.tool_delays)} tool delays; there should be one fewer delay"
            )

    gpus = [GPU(index, speed) for index, speed in enumerate(speeds)]
    prepared: List[List[_Prepared]] = [_prepare(program) for program in programs]
    completion: List[Optional[float]] = [None] * len(programs)

    # (time, kind, sequence, payload). sequence makes the heap deterministic
    # when two events of the same kind share a timestamp: lower sequence, which
    # is the order we scheduled them, wins.
    events: List[Tuple[float, int, int, object]] = []
    sequence = 0

    def push(time: float, kind: int, payload: object) -> None:
        nonlocal sequence
        heapq.heappush(events, (time, kind, sequence, payload))
        sequence += 1

    for program_index, program in enumerate(programs):
        # payload for a ready event: which program, which call, which GPU ran the previous call
        push(program.arrival_time, _READY, (program_index, 0, None))

    while events:
        now, kind, _, payload = heapq.heappop(events)
        if kind == _FINISH:
            gpu_index = payload
            gpu = gpus[gpu_index]
            finished = gpu._take_finished(now)
            program = programs[finished.program_index]
            info = prepared[finished.program_index][finished.call_index]
            last_call = finished.call_index + 1 == len(program.calls)
            if last_call:
                # Nothing will reuse this prefix. Writing it and then deleting
                # it would evict some other program's cache on the way through,
                # so we only drop what this program left behind on earlier calls.
                for each in gpus:
                    each._drop(program.id)
                completion[finished.program_index] = now - program.arrival_time
            else:
                # The next prompt starts with this prompt plus the tokens just
                # generated. Keep that prefix. If it does not fit, the least
                # recently written program on this GPU is evicted.
                gpu._set_cache(program.id, info.context_tokens + info.decode_tokens)
                delay = program.tool_delays[finished.call_index]
                push(
                    now + delay,
                    _READY,
                    (finished.program_index, finished.call_index + 1, gpu.id),
                )
            # Now the cache matches reality. The call that has been waiting
            # longest may start, and it prices its prefill against this cache.
            next_finish = gpu._start_next(now)
            if next_finish is not None:
                push(next_finish, _FINISH, gpu_index)
            continue

        program_index, call_index, prev_gpu_id = payload
        program = programs[program_index]
        info = prepared[program_index][call_index]
        call = _make_call(program, call_index, info, prev_gpu_id)
        chosen = router.pick(call, gpus, now)
        gpu = _resolve_gpu(chosen, gpus)
        job = _Job(program_index, call_index, call, info.decode_tokens)
        finish_at = gpu._enqueue(job, now)
        if finish_at is not None:
            push(finish_at, _FINISH, gpu.id)

    missing = [programs[i].id for i, value in enumerate(completion) if value is None]
    if missing:
        raise RuntimeError(f"simulation ended with unfinished programs: {missing}")
    return [float(value) for value in completion]  # type: ignore[arg-type]


def _resolve_gpu(chosen: object, gpus: List[GPU]) -> GPU:
    """pick() should return one of the GPU objects it was given. Match by identity, then by id."""
    for gpu in gpus:
        if chosen is gpu:
            return gpu
    chosen_id = getattr(chosen, "id", None)
    for gpu in gpus:
        if gpu.id == chosen_id:
            return gpu
    raise RuntimeError(
        "router.pick returned something that is not one of the GPUs it was given"
    )


def p95(values: List[float]) -> float:
    """95th percentile, linear interpolation (NumPy's default).

    For a sorted sample x[0] .. x[n-1], the percentile sits at position
    (n - 1) * 0.95 and interpolates between the two surrounding samples.
    An empty list has no 95th percentile.
    """
    if not values:
        raise ValueError("p95 needs at least one value")
    return float(np.percentile(np.asarray(values, dtype=float), 95))
