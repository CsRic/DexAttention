"""Shared definitions of the offline traffic and balance statistics (Table 4, imbalance / NIC / direct-fetch columns).

Traffic record = one (system, workload, case): recv[pass]["intra" | "inter"] = bytes received by rank 0..P-1, where
intra = the source GPU is on the receiver's node. The NIC load of a case is the largest per-node sum of inter-node
receives (one NIC per node). Every input path is resolved here, so the data layout is documented in one place.
Nothing in this directory uses a GPU unless a script says so (w5_comm.py --gpu, direct_fetch.py w5).
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dex.common.env import cfg  # noqa: E402

P = 16              # GPUs
GPN = 4             # GPUs per node (rank r is on node r // GPN)
TILE, D = 128, 128  # tile edge, head dimension
MiB = float(1 << 20)


def data_root():
    return cfg("DEX_DATA_ROOT")


def result_root():
    return cfg("DEX_RESULT_ROOT")


def out_dir(*sub):
    d = os.path.join(result_root(), "comm_volume", *sub)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- input layout
def _static():
    """experiments/attention_layer/static_inputs.py: the single definition of the W1-W3 inputs and file locations"""
    d = os.path.join(REPO, "experiments", "attention_layer")
    if d not in sys.path:
        sys.path.insert(0, d)
    import static_inputs
    return static_inputs


def static_cases(W):
    """{input id: case dict (id, mask, ranges, S)} of W1 / W2 / W3 in canonical order"""
    return {c["id"]: c for c in _static().inputs(W.lower())}


def static_names(W):
    return list(static_cases(W))


def mask_path(W, name):
    """tile mask npz (mask [n, n] int8, heads, tokens) of a static input"""
    return static_cases(W)[name]["mask"]


def ranges_path(W, name):
    """per-token key ranges [S, R, 2] of the same mask (real tokens only)"""
    return static_cases(W)[name]["ranges"]


def magi_input(W, name):
    """Magi's slices of a static input: W1 token ranges, W2 / W3 the rollout segment table of the 64-block mask"""
    if W == "W1":
        return dict(ranges=ranges_path(W, name))
    return dict(segs=mask_path(W, name).replace("_nb32.npz", "_nb64.npz"))


def dex_plan_paths(W, name):
    """head_split=4 Dex plans: block-diagonal split mask + forward / backward plan JSON (owner, schedule_table, ...)"""
    return _static().plan_files(static_cases(W)[name], 4)


def ua_dir():
    """UA plan directory: manifest.json (per input and pass: chosen block size and plan file) + plan JSONs"""
    return os.path.join(data_root(), "plans", "ua")


def dcp_dir(W):
    """DCP plan directory of a workload: ladder.json (final block size per input) + cache/<name>_B<B>.cp<rank>.pkl"""
    return os.path.join(data_root(), "plans", "dcp", W.lower())


def w4_selection_dir():
    return os.path.join(data_root(), "w4_infllm", "masks")


def w4_cases():
    """W4 operator-test cases '<seq>:<layer>' (36 sequences x layers 0 / 8 / 16 / 24)"""
    return read_names(os.path.join(REPO, "workloads", "w4_infllm", "cases144.txt"))


def w5_cases():
    """W5 cases: (dump directory, call index, case name), the 42 cases of workloads/w5_sparge/cases42.txt"""
    from workloads.w5_sparge.cases import dump_dir, read_cases
    return [(dump_dir(g), c, f"{g}_c{c:05d}") for g, c in read_cases()]


def w6_cases():
    """W6 cases '<input>:<layer>' (9 inputs x layers 0 / 8 / 16 / 24)"""
    return [tuple(c.split(":")) for c in read_names(os.path.join(REPO, "workloads", "w6_flexprefill", "cases36.txt"))]


def w6_dump(inp, layer, what):
    """what = mask (bit-packed [32, nb, nb] uint8 FlexPrefill tile mask) | q"""
    suffix = "mask_g095" if what == "mask" else what
    return os.path.join(data_root(), "w6_flexprefill", "qkv", inp, f"L{int(layer):02d}_{suffix}.npy")


def attention_records(W, system, launch=1):
    """operator-test records of a dynamic workload (one JSON object per case)"""
    f = os.path.join(result_root(), "attention_layer", W.lower(), system, f"rep{launch}.jsonl")
    return [json.loads(x) for x in open(f) if x.strip()] if os.path.exists(f) else []


# ---------------------------------------------------------------- accounting
def node(r):
    return r // GPN


def empty_recv():
    return {ps: {"intra": [0] * P, "inter": [0] * P} for ps in ("fwd", "bwd")}


def add(recv, ps, src, dst, nbytes):
    """count nbytes received by dst from src in pass ps (src == dst is a local copy and is not counted)"""
    if src == dst or nbytes == 0:
        return
    recv[ps]["intra" if node(src) == node(dst) else "inter"][dst] += int(nbytes)


def nic_max(inter):
    """largest per-node sum of inter-node receives, bytes"""
    return max(sum(inter[n * GPN:(n + 1) * GPN]) for n in range(P // GPN))


def zigzag_owner(n_chunks_total, P_=P):
    """owner rank of each of 2P equal chunks: rank r holds chunks r and 2P - 1 - r"""
    return [c if c < P_ else 2 * P_ - 1 - c for c in range(n_chunks_total)]


def write_jsonl(path, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def read_names(path):
    return [x.strip() for x in open(path) if x.strip() and not x.startswith("#")]


def assert_no_cuda():
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "run with CUDA_VISIBLE_DEVICES= (empty): CPU only"
