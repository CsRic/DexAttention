"""UA's op for W2 / W3 training (experiments/end2end/qwen_train.py): UA's forward plan and the backward plan on the
forward's deployment, from the manifest of baselines/ua/planner.py, on the plan's own block layout (--masks of the driver)."""
import numpy as np

from baselines.ua.static import UAStatic, manifest


def make_e2e_op(d):
    """d: id, mask_path, mask, heads, pad, fwd_plan, bwd_plan, workload; plans default to the manifest's"""
    man = manifest()

    def plan(given, ps):
        if given:
            return given
        ent = man.get(f"{d['id']}|{ps}", dict(na="no plan in the manifest"))
        if "na" in ent:                                # the planner's recorded reason reaches the log of the launch
            raise RuntimeError(f"UA has no {ps} plan for {d['id']}: {ent['na']}")
        return ent["plan"]
    fwd, bwd = plan(d.get("fwd_plan"), "fwd"), plan(d.get("bwd_plan"), "bwd_on_fwd")
    return UAStatic(np.asarray(d["mask"], np.int8), fwd, bwd, d["heads"], pad=d["pad"])
