"""Dex planner (assigner): maps every block pair of a sparse attention mask to a GPU, node level then GPU level.

    from dex.planner import Planner, plan, plan_blocks, bwd_models_b, to_blocks, FWD_MODEL

  plan(W, owner, gpn, NN, Qb, Ob, Kb, TT, bwd=None)   full [n, n] W -> GPU of every block pair (offline plan files)
  Planner(H, m, gpn, NN, Qb, Ob, Kb, TT, models)      compiled (torch.compile + CUDA graphs) per-head-block planner;
                                                      call(Wb, ob, gid) -> plan [H, m, m] (online)
  plan_blocks(...)                                    eager reference implementation of Planner's algorithm (same rounds and rules; see plan.py)
  bwd_models_b(heads, B, kb)                          backward time models (models= for the backward plan)
  TT (seconds per tile, forward) = FWD_MODEL["T_TILE"] * hq / 32; (Qb, Ob, Kb) = fwd_bytes((hq, hk), B) (fp32 partial O).
Modules: cluster (Q-KV clustering), kv (KV re-split), levels (node / GPU level), models (time models), ops (batched
reductions), placement (owner placement candidates), plan (drivers)."""
from .models import FWD_MODEL, BWD_MODEL, fwd_bytes, bwd_bytes, bwd_models_b
from .plan import Planner, plan, plan_blocks, head_blocks, to_blocks
from .placement import zigzag_equal, contiguous_equal, choose as choose_placement  # noqa: F401

