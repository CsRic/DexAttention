"""Schedule types of the decomposed executor: a per-rank list of global steps; every comm op is an explicit
point-to-point edge (block, src, dst), every comp op a rectangle (q blocks x kv blocks)."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class CommOp:
    blocks: List[int]        # global block ids moved together
    src: int
    dst: int
    io: str                  # 'i' (Q / KV input) or 'o' (partial result back to the owner)
    rc: str                  # 'r' (row: Q or O) or 'c' (column: KV)

    def is_send(self, rank):
        return rank == self.src

    def is_recv(self, rank):
        return rank == self.dst


@dataclass
class CompOp:
    q_blocks: List[int]
    kv_blocks: List[int]
    fused: bool = False


@dataclass
class Step:
    number: int = 0
    comm: List[CommOp] = field(default_factory=list)
    comp: List[CompOp] = field(default_factory=list)


@dataclass
class FusionSchedule:
    world: int
    rank: int
    n_blocks: int
    block_tokens: int
    owner: List[int]                                     # owner[block] -> rank
    steps: List[Step] = field(default_factory=list)
    init_q: List[int] = field(default_factory=list)      # blocks resident on this rank at the start
    init_kv: List[int] = field(default_factory=list)
    owned_q: List[int] = field(default_factory=list)     # q blocks whose final merged O this rank holds
