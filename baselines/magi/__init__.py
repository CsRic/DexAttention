"""MagiAttention baseline: Magi's own dispatch, solver and executor (third_party/magiattention, third_party/PINS.md)
with its all_to_all replaced by the staged exchange (staged_a2a.py). FA4 kernel backend; fp32 backward partial reduce."""
import os
import typing

if not hasattr(typing, "override"):    # Magi targets Python 3.12 typing
    typing.override = lambda f: f

os.environ.setdefault("MAGI_ATTENTION_KERNEL_BACKEND", "fa4")
os.environ.setdefault("MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE", "1")
