"""W5 data generation: the 32 Wan2.1-T2V-1.3B generations (8 VBench prompts x seeds {0, 1} x {480p, 720p}, 81 frames,
50 denoising steps) with SGLang Diffusion (Ulysses sequence parallelism over 4 GPUs of one node, SpargeAttn backend,
top-k 0.3), recording q / k of the conditional layers 0 / 15 / 29 at steps {0, 5, 15, 25, 35, 45, 49} (dump_hook.py) and
the generated video (input of prep_e2e_inputs.py).

Inputs : prompts.json (VBench @ fd18b3d, per dimension a numpy default_rng(0) permutation, first 2 distinct prompts),
         $DEX_MODEL_ROOT/Wan2.1-T2V-1.3B-Diffusers
Outputs: $DEX_DATA_ROOT/w5_sparge/dumps/<res>_p<i>_s<seed>/rank<r>/call<i>.npz (~7 GB per generation; node-local
         storage recommended), $DEX_DATA_ROOT/w5_sparge/videos/<res>_p<i>_s<seed>.mp4
Run inside an environment with SGLang Diffusion and SpargeAttn installed (`sglang` on PATH), one node with 4 GPUs.
The jobs are dealt round-robin over --num-nodes; each node runs its share sequentially.
usage: python generate.py --node-index 0 --num-nodes 1 [--no-dump]"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dex.common.env import cfg  # noqa: E402

RES = {"480p": (480, 832), "720p": (720, 1280)}
FRAMES, STEPS, TOPK, ULYSSES = 81, 50, 0.3, 4
CALLS_PER_STEP = 60  # 30 layers x (conditional, unconditional)
DUMP_STEPS = "0,5,15,25,35,45,49"
DUMP_QK = "0,15,29"


def sglang_cmd(model, H, W, seed, prompt, out):
    return [
        "sglang",
        "generate",
        "--model-path",
        model,
        "--num-gpus",
        str(ULYSSES),
        "--ulysses-degree",
        str(ULYSSES),
        "--tp-size",
        "1",
        "--enable-cfg-parallel",
        "false",
        "--encoder-parallel",
        "replicate",
        "--vae-config.use-parallel-decode",
        "false",
        "--vae-config.use-tiling",
        "true",
        "--dit-cpu-offload",
        "false",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--attention-backend",
        "sparge_attn",
        "--attention-backend-config",
        f"topk={TOPK}",
        "--height",
        str(H),
        "--width",
        str(W),
        "--num-frames",
        str(FRAMES),
        "--num-inference-steps",
        str(STEPS),
        "--seed",
        str(seed),
        "--prompt",
        prompt,
        "--output-path",
        out,
        "--save-output",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-index", type=int, default=0)
    ap.add_argument("--num-nodes", type=int, default=1)
    ap.add_argument("--no-dump", action="store_true", help="video only (no q / k dump)")
    a = ap.parse_args()
    man = json.load(open(os.path.join(HERE, "prompts.json")))
    model = os.path.join(cfg("DEX_MODEL_ROOT"), "Wan2.1-T2V-1.3B-Diffusers")
    root = os.path.join(cfg("DEX_DATA_ROOT"), "w5_sparge")
    vids = os.path.join(root, "videos")
    os.makedirs(vids, exist_ok=True)
    jobs = [(res, i, s) for res in ("720p", "480p") for i in range(len(man["prompts"])) for s in man["seeds"]]
    mine = [j for n, j in enumerate(jobs) if n % a.num_nodes == a.node_index]
    for res, i, s in mine:
        tag = f"{res}_p{i}_s{s}"
        dump = os.path.join(root, "dumps", tag)
        if os.path.exists(f"{dump}/DONE") and os.path.exists(f"{vids}/{tag}.mp4"):
            continue
        H, W = RES[res]
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        env["PYTHONPATH"] = os.path.join(HERE, "dump_site") + os.pathsep + env.get("PYTHONPATH", "")
        if not a.no_dump:
            env.update(
                W5_DUMP_DIR=dump,
                W5_DUMP_CALLS_PER_STEP=str(CALLS_PER_STEP),
                W5_DUMP_STEPS=DUMP_STEPS,
                W5_DUMP_QK=DUMP_QK,
            )
        out = os.path.join(cfg("DEX_SCRATCH"), "w5_generate", tag)
        t0 = time.time()
        r = subprocess.run(sglang_cmd(model, H, W, s, man["prompts"][i]["prompt"], out), env=env)
        mp = sorted(glob.glob(f"{out}/**/*.mp4", recursive=True))
        if mp:
            shutil.copy(mp[0], f"{vids}/{tag}.mp4")
        ncall = len(os.listdir(f"{dump}/rank0")) if os.path.isdir(f"{dump}/rank0") else 0
        if not a.no_dump and r.returncode == 0 and ncall == 7 * CALLS_PER_STEP:
            open(f"{dump}/DONE", "w").write("ok\n")
        print(
            json.dumps(
                dict(tag=tag, rc=r.returncode, calls_rank0=ncall, video=bool(mp), seconds=round(time.time() - t0, 1))
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
