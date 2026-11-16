"""Schedule stage + kernel tables of the distributed sparse-attention executor, and the forward launch.

  fwd     block_pairs / block_pairs_heads, schedule_blocks (global, gateway mode 5), tables_blocks (this PE)
  bwd     build_fast (schedule + this PE's backward tables), region_layout, scratch_layout, TABLE_ORDER, RECT_ORDER
  local   per-rank planning (enabled(P): DIST_LOCAL_PLAN, auto = from 64 GPUs): fwd_tables, bwd_tables (gateway mode 3)
  launch  load_plan, call (dex_cuda.dist_fwd with the engine's message tables)
  xorder  another system's execution order imposed on these tables (component-replacement study, Table 5)
C++: csrc/fastsched_blocks.cpp (lib/libfsched.so), csrc/bwdplan.cpp (lib/libbwdplan.so), built by build.sh."""
from . import fwd, bwd, local, launch
from .fwd import LINK, block_pairs, block_pairs_heads, block_pairs_heads_torch, schedule_blocks, tables_blocks
from .bwd import build_fast, region_layout, scratch_layout, expand_plan, TABLE_ORDER, RECT_ORDER
from .launch import load_plan, call
