"""Shared contract between Person 1's engine (workload + simulator) and Person 2's routers.
Don't change this file without telling the other person."""
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

# Fast-GPU throughput for an 8B model (batched). Slow GPU = these / speed ratio.
PREFILL_RATE = 10_000   # tokens per second
DECODE_RATE = 1_500     # tokens per second
CACHE_TOKENS = 300_000  # KV cache capacity per GPU, in tokens
DEFAULT_DECODE_GUESS = 34

@dataclass
class Program:
    id: int
    arrival_time: float
    calls: List[Tuple[int, int]]   # (new_prompt_tokens, decode_tokens) per call
    tool_delays: List[float]       # seconds between consecutive calls; len = len(calls) - 1

@dataclass
class Call:
    program_id: int
    index: int                     # which call in the program (0, 1, 2, ...)
    context_tokens: int            # full prompt length, including the program's history
    new_tokens: int                # tokens added since the previous call
    decode_guess: float            # estimate only; routers never see the true decode length
    prev_gpu_id: Optional[int]     # GPU that ran this program's previous call (None for the first)

class GPUView(Protocol):
    """What a router is allowed to see about a GPU."""
    id: int
    speed: float                   # 1.0 = fast GPU; slow GPU = 1 / ratio
    def cached(self, program_id: int) -> int: ...      # tokens of this program cached here
    def outstanding_calls(self) -> int: ...            # calls waiting + running
    def tokens_in_flight(self) -> int: ...             # estimated uncached prefill + decode tokens queued/running
    def est_seconds_left(self, now: float) -> float: ...  # estimated seconds until this GPU's queue drains

class Router(Protocol):
    name: str
    def pick(self, call: Call, gpus: List[GPUView], now: float) -> GPUView: ...