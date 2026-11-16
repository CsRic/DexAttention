"""The mask sets of the rollout and W4 series of plan_scaling.py, and where their tile masks live.

workloads/w2_w3_rollout/lengths/rollout_groups.jsonl
                      one Qwen3-4B rollout group per line: id (<star|tree>_<split>_p<prompt>), workloads, P (prompt tokens),
                      nodes [[reply tokens, parent node or -1 for the prompt], ...]. Token counts only: a tile mask depends on
                      nothing else. W2 and W3 share these masks and differ in their heads.
w4_masks.txt          '<sequence>:<layer>' of the 5 W4 masks.
The order of each list is the order the seeded packing permutes, so it is part of the experiment's definition.
Inputs : the two lists. Outputs: none (imported by gen_family_masks.py and plan_scaling.py).
Example: from family_lists import rollout_groups
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def rollout_groups(workload=None):
    """rollout group records in list order, optionally only those of workload (w2 | w3)"""
    with open(os.path.join(HERE, "..", "..", "workloads", "w2_w3_rollout", "lengths", "rollout_groups.jsonl")) as f:
        rows = [json.loads(x) for x in f if x.strip()]
    return [r for r in rows if workload is None or workload in r["workloads"]]


def w4_cases():
    """[(sequence, layer)] in list order"""
    with open(os.path.join(HERE, "w4_masks.txt")) as f:
        lines = [x.strip() for x in f if x.strip() and not x.startswith("#")]
    return [(c.split(":")[0], int(c.split(":")[1])) for c in lines]


def rollout_mask_file(mask_dir, group_id):
    return os.path.join(mask_dir, "rollout", f"{group_id}.npz")


def w4_mask_file(mask_dir, seq, layer):
    return os.path.join(mask_dir, f"w4_{seq}_L{layer:02d}.npz")
