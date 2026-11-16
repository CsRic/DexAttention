"""UltraAttn (UA) baseline: UA's planner (deployment ILPs, fusion and schedule; deploy.py, planner.py, third_party/ultraattn)
and the decomposed executor its plans run on (pool_exec.py, pool_exec_bwd.py, on host-launched NCCL point-to-point)."""
import os
import sys

_FA = os.environ.get("DEX_FA283_PATH")   # the official flash-attn 2.8.3 build, first on the path for the executor's kernels
if _FA and _FA not in sys.path:
    sys.path.insert(0, _FA)
