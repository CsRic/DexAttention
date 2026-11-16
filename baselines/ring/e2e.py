"""Ring's op for W2 / W3 training (experiments/end2end/qwen_train.py): cp_attention interface on the zigzag layout."""
import numpy as np

from baselines.ring.static import RingStatic


def make_e2e_op(d):
    """d: id, mask_path, mask, heads, pad, fwd_plan, bwd_plan, workload"""
    return RingStatic(np.asarray(d["mask"], np.int8), d["heads"], pad=d["pad"])
