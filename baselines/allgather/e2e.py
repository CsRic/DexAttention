"""All-gather's op for W2 / W3 training (experiments/end2end/qwen_train.py): cp_attention interface on the zigzag layout."""
import numpy as np

from baselines.allgather.static import AllGatherStatic


def make_e2e_op(d):
    """d: id, mask_path, mask, heads, pad, fwd_plan, bwd_plan, workload"""
    return AllGatherStatic(np.asarray(d["mask"], np.int8), d["heads"], pad=d["pad"])
