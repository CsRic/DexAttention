"""UA (UltraAttn plans on the decomposed executor in baselines/ua) on W1-W3: per-GPU received bytes from the plan JSONs.

Plan JSON: block_tokens B, owner[N], schedule_table[N][N], gpu_kernels[g] = [{kind: comm, block, src, dst, io: i|o,
rc: r|c, role: send|recv} | {kind: comp, ...}]. The executor sends one point-to-point transfer per tensor per comm record,
whole blocks (all heads, B tokens), no relay. Bytes per block (H = query heads, Hk = KV heads, D = 128):
  forward : i,r Q fp16 2BHD | i,c K+V fp16 4BHkD | o,r O fp32 4BHD + LSE fp32 4HB
  backward: i,r Q+dO fp16 4BHD + Delta+LSE fp32 8HB | i,c K+V fp16 4BHkD | o,r dQ fp32 4BHD | o,c dK+dV fp32 8BHkD
Checks: the receive-side records equal the send-side records and the matrix rebuilt from schedule_table + owner.
Inputs : common.ua_dir()/manifest.json = {"<input>|fwd" | "<input>|bwd": {"plan": <file>} or {"na": <reason>}}
Output : $DEX_RESULT_ROOT/comm_volume/recv/ua_<W>.jsonl
Example: CUDA_VISIBLE_DEVICES= python experiments/comm_volume/ua_comm.py
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def per_block(B, H, Hk, bwd):
    D = C.D
    if not bwd:
        return {("i", "r"): 2 * B * H * D, ("i", "c"): 4 * B * Hk * D, ("o", "r"): 4 * B * H * D + 4 * H * B}
    return {("i", "r"): 4 * B * H * D + 8 * H * B, ("i", "c"): 4 * B * Hk * D, ("o", "r"): 4 * B * H * D,
            ("o", "c"): 8 * B * Hk * D}


def count(plan, H, Hk):
    d = json.load(open(plan))
    B, bwd = d["block_tokens"], d.get("fob") == 1
    assert d["cp"] == C.P and d["gpus_per_node"] == C.GPN, plan
    pb = per_block(B, H, Hk, bwd)
    M, msend = Counter(), Counter()
    for g, L in enumerate(d["gpu_kernels"]):
        for k in L:
            if k["kind"] != "comm":
                continue
            key = (k["io"], k["rc"])
            if k["dst"] == g and k.get("role", "recv") == "recv":
                M[(k["src"], g, key)] += pb[key]
            if k["src"] == g and k.get("role", "send") == "send":
                msend[(k["src"], k["dst"], key)] += pb[key]
    st, own = d["schedule_table"], d["owner"]
    N = len(own)
    R = Counter()
    for g in range(C.P):
        rows = {q for q in range(N) for kv in range(N) if st[q][kv] == g}
        cols = {kv for q in range(N) for kv in range(N) if st[q][kv] == g}
        for q in rows:
            if own[q] != g:
                R[(own[q], g, ("i", "r"))] += pb[("i", "r")]; R[(g, own[q], ("o", "r"))] += pb[("o", "r")]
        for kv in cols:
            if own[kv] != g:
                R[(own[kv], g, ("i", "c"))] += pb[("i", "c")]
                if bwd:
                    R[(g, own[kv], ("o", "c"))] += pb[("o", "c")]
    return dict(B=B, bwd=bwd, M=M, consistent_send=(M == msend), consistent_table=(M == R))


def main():
    C.assert_no_cuda()
    man = json.load(open(os.path.join(C.ua_dir(), "manifest.json")))
    for W in ("W1", "W2", "W3"):
        recs = []
        for n in C.static_names(W):
            rec = dict(system="UA", workload=W, case=n)
            recv = C.empty_recv(); have = {}
            H, Hk = (32, 8) if W == "W3" else (16, 16)
            for ps in ("fwd", "bwd"):
                e = man.get(f"{n}|{ps}") or {"na": "not in the manifest"}
                plan = os.path.join(C.ua_dir(), e["plan"]) if "plan" in e else None
                if plan is None or not os.path.exists(plan):
                    have[ps] = e.get("na", f"plan file missing {plan}"); continue
                c = count(plan, H, Hk)
                assert c["bwd"] == (ps == "bwd"), plan
                for (s, d_, _key), b in c["M"].items():
                    C.add(recv, ps, s, d_, b)
                have[ps] = dict(plan=os.path.basename(plan), B=c["B"], send_recv_consistent=c["consistent_send"],
                                table_consistent=c["consistent_table"])
            rec["plans"] = have
            rec["passes"] = [ps for ps in ("fwd", "bwd") if isinstance(have[ps], dict)]
            if rec["passes"]:
                rec["recv"] = recv
            else:
                rec["na"] = "; ".join(f"{ps}: {have[ps]}" for ps in ("fwd", "bwd"))
            recs.append(rec)
        C.write_jsonl(os.path.join(C.out_dir("recv"), f"ua_{W}.jsonl"), recs)
        print(f"[ua] {W}: {sum('recv' in r for r in recs)}/{len(recs)} cases with a plan")


if __name__ == "__main__":
    main()
