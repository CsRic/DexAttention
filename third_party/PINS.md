# Third-party pins

`bash third_party/setup_submodules.sh [names]` fetches each repository below at its commit and applies its change file from `build/patches/` (`git apply` at the repository root; the flash-attention and InfLLM-v2 CUTLASS change files are applied by `build/build_ext.sh` instead). The URLs and branches are in `.gitmodules` at the root of the tree. Use this script rather than a recursive submodule update: UltraAttn's nested `third_party/kernel_profiler` is not public, and only the nested repositories named here are needed.

| path under `third_party/` | repository | commit | change file | purpose | licence |
|---|---|---|---|---|---|
| `flash-attention` | https://github.com/Dao-AILab/flash-attention | `6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15` (tag `fa4-v4.0.0.beta15`) | `flash-attention.patch`, by `build/build_ext.sh` | headers Dex's kernels compile against (`hopper/`, `csrc/flash_attn/`) | BSD-3-Clause |
| `flash-attention/csrc/cutlass` (nested) | https://github.com/NVIDIA/cutlass | `7127592069c2fe01b041e174ba4345ef9b279671` | none | CUTLASS of Dex's extension | BSD-3-Clause |
| `infllmv2_cuda_impl` | https://github.com/OpenBMB/infllmv2_cuda_impl | `93cf2ec28e5a7acebe3f0bb7329b6c73a1be91f6` | none | InfLLM-v2 kernel of W4: headers for Dex's routines, `infllm_v2` package | Apache-2.0, stated in its overview; no licence file |
| `infllmv2_cuda_impl/csrc/cutlass` (nested) | https://github.com/zhangyan-didu/cutlass (branch `infllmv2-cuda13`) | `424c5a03220f58a56ddd754e0e2d4eabdf01c802` | `infllmv2_cutlass.patch`, by `build/build_ext.sh` | CUTLASS of `infllm_v2.C` only | BSD-3-Clause |
| `yunchang` | https://github.com/feifeibear/long-context-attention | `a5f8f600d435ced180a97c91c68e87ff3cfb8f89` | none | Ring: `RingComm`, `update_out_and_lse` | Apache-2.0 |
| `ultraattn` | https://github.com/oliverYoung2001/UltraAttn | `b5aa10e18f996de59f5237c00ae9094cf911b9c7` | `ultraattn.patch` | UA deployment, fusion and schedule stages | Apache-2.0 |
| `dcp` | https://github.com/chenyu-jiang/dcp | `e1e19b8222349bb98d632016dc7544e26e6bcdac` | `dcp.patch` | DCP planner and executor | MIT |
| `dcp_flash_attention` | https://github.com/chenyu-jiang/flash-attention (branch `dcp`) | `a15bcc7cac1a854a77677e90118fedfe48533a2e` | `dcp_flash_attention.patch` | DCP's kernels | BSD-3-Clause |
| `dcp_flash_attention/csrc/cutlass` (nested) | https://github.com/NVIDIA/cutlass | `756c351b4994854b2f8c6dded3821ebbb580876b` | none | CUTLASS of DCP's kernels | BSD-3-Clause |
| `magiattention` | https://github.com/SandAI-org/MagiAttention | `a3276e7bf06027f1bac540e50187cbbc798517f4` | `magiattention.patch` | Magi planner and executor | Apache-2.0 |
| `magiattention/magi_attention/functional/flash-attention` (nested) | https://github.com/jiayus-nvidia/flash-attention (branch `magi_backend`) | `9e38ce29823a150aa979ea8efdaee67c9ee6d62b` | `magiattention_flash_attention.patch` | Magi's FA4 backend (`flash_attn_cute`); its `csrc/cutlass` is the same `7127592` | BSD-3-Clause |
| `flex_prefill` | https://github.com/bytedance/FlexPrefill | `a24b27bf117fd6ad4ba34b0db82c598e7786fd59` | none | W6 block selection, official reference | Apache-2.0 |
| `SpargeAttn` | https://github.com/thu-ml/SpargeAttn | `ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a` | `spargeattn.patch` | W5 generations and selection (`spas_sage_attn`) | Apache-2.0 |

## Which FlashAttention is which

- **flash-attention `6c4f74f`** (above, with `flash-attention.patch`): only headers. Dex's kernels compile against them; nothing imports this tree as a Python package.
- **FlashAttention 2.8.3** (tag `v2.8.3`, `060c9188beec3a8b62b33a3bfa6d5d2d44975fab`, its CUTLASS `dc4817921edda44a549197ff3a9dcf5df0636e7b`): the standalone FA2 routine of the baselines. UA's decomposed executor and the microbenchmark import it, found through `DEX_FA283_PATH`. It is a separate checkout, built with the core environment's Python:

  ```bash
  # Set DEX_FA283_PATH to a new checkout path and DEX_CUDA_HOME to CUDA 12.1.
  git clone -b v2.8.3 --depth 1 --recurse-submodules=csrc/cutlass --shallow-submodules https://github.com/Dao-AILab/flash-attention.git "$DEX_FA283_PATH"
  (cd "$DEX_FA283_PATH" && CUDA_HOME="$DEX_CUDA_HOME" FLASH_ATTN_CUDA_ARCHS=80 FLASH_ATTENTION_FORCE_BUILD=TRUE "$DEX_PYTHON" setup.py build_ext --inplace)
  ```
- **InfLLM-v2** (`infllmv2_cuda_impl`): W4's kernel, with its own FlashAttention-derived sources and its own CUTLASS.
- **DCP's fork** and **Magi's fork** of flash-attention belong to those baselines only.

## Other dependencies

| dependency | version | used by |
|---|---|---|
| NVSHMEM | 3.7.1 (`DEX_NVSHMEM_HOME`) | Dex's extension and runtime |
| PyTorch for DCP | https://github.com/chenyu-jiang/pytorch, branch `alltoallv`, commit `942739513579312485303ff3dd961a72cd97579e` (v2.5.1 with `all_to_all` skipping zero-byte send / recv), licence BSD-3-Clause | DCP's environment; built with `python setup.py bdist_wheel` for CUDA 12.1 and the pip NCCL 2.21.5 (`USE_SYSTEM_NCCL=1`, `_GLIBCXX_USE_CXX11_ABI=0`, `USE_MPI=0`, `TORCH_CUDA_ARCH_LIST=8.6`), then `pip install --no-deps --force-reinstall` |
| pypatoh | https://github.com/chenyu-jiang/pypatoh, commit `884317e3d18dc1c1c7c7556b1edd45556b0817d1`, with the PaToH library (`PATOH_PATH`); licence not stated in the repository. PaToH is not part of it: obtain it from its authors under their terms | DCP's planner imports it |
| redis-server | 8.10.2 (`DEX_REDIS_SERVER`) | DCP's data loader |
| Gurobi | `gurobipy` 12.0.3 with a licence (`DEX_GUROBI_LICENSE`) | UA's deployment and schedule ILPs |
| vLLM | https://github.com/vllm-project/vllm, commit `bc39ded3c4e7b0453af76c617883e5cd88092222`, Apache-2.0 | W2 / W3 rollouts |
| SGLang | `sglang` 0.5.21, `sglang-kernel` 0.4.6.post1, Apache-2.0 | W5 generations |

Python packages per environment are pinned in `requirements/`.

## Installing Magi

Install the tree into Magi's environment (`pip install --no-build-isolation third_party/magiattention`), so that the imported copy carries the changes. Build its FA4 submodule with `FA_NO_DOWNLOAD=1`, install it editable, and put `csrc/utils/create_block_mask` and `csrc/utils/magi_to_hstu` on the Python path (a `.pth` file in site-packages).
