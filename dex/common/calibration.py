"""Cost-model constants of the planner and scheduler, read once from config/calibration.json (or the file named by
DEX_CALIBRATION). The JSON stores each constant in the unit named by its key suffix; this module converts them to
the units the models use (seconds, bytes per second)."""
import json
import os

from .env import repo_root

GiB = float(1 << 30)
_CACHE = None


def load() -> dict:
    """the parsed calibration file (cached)"""
    global _CACHE
    if _CACHE is None:
        path = os.environ.get("DEX_CALIBRATION") or os.path.join(repo_root(), "config", "calibration.json")
        with open(path) as f:
            _CACHE = json.load(f)
    return _CACHE


def fwd_model() -> dict:
    """forward time-model constants: seconds, bytes per second"""
    c = load()["forward"]
    return dict(T_TILE=c["t_tile_s"], A_COMP=c["a_comp"], GAMMA=c["gamma"], C_END=c["c_end_s"],
                B_NIC=c["b_nic_gib_s"] * GiB, KAPPA=c["kappa"], B_INTRA=c["b_intra_gib_s"] * GiB, C_ARR=c["c_arr_s"])


def bwd_model() -> dict:
    """backward time-model constants: seconds, bytes per second"""
    c = load()["backward"]
    return dict(k_comp=c["k_comp"], B_eng=c["b_eng_b_s"], B_nic=c["b_nic_b_s"], kappa=c["kappa"], c0=c["c0_s"])


def bwd_coef() -> dict:
    """{(query heads, KV heads): (seconds per task, seconds per item)} of the backward compute profile"""
    c = load()["backward"]; n = c["profile_count"]
    out = {}
    for k, v in c["coef"].items():
        hq, hk = (int(x) for x in k.split("x"))
        out[(hq, hk)] = (v["task_s"] / n, v["item_s"] / n)
    return out


def infllm_entry_times() -> dict:
    """InfLLM-v2 planner: seconds per entry of the official forward / backward routine"""
    c = load()["infllm"]
    return dict(T_ENTRY_FWD=c["t_entry_fwd_s"], T_ENTRY_BWD=c["t_entry_bwd_s"])


def links() -> dict:
    """scheduler link constants: seconds, bytes per second"""
    c = load()["links"]
    return dict(BW_INTER=c["bw_inter_gib_s"] * GiB, BW_INTRA=c["bw_intra_gib_s"] * GiB, BW_FW=c["bw_fw_gib_s"] * GiB,
                LAT_INTER=c["lat_inter_s"], LAT_INTRA=c["lat_intra_s"])
