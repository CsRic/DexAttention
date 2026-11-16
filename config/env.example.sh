# Machine-specific configuration. Copy to config/env.sh, fill in, and every script sources it.
# Cluster
export DEX_HOSTFILE=${DEX_HOSTFILE:-config/hostfile.example}  # one hostname per line, node 0 first (relative to the checkout or absolute)
export DEX_MASTER_ADDR=${DEX_MASTER_ADDR:-node0}              # torchrun rendezvous address: node 0's address on the interface below
export DEX_MASTER_PORT=${DEX_MASTER_PORT:-29500}              # torchrun rendezvous port
export DEX_GPUS_PER_NODE=${DEX_GPUS_PER_NODE:-4}              # GPUs per node
export DEX_NIC_IFNAME=${DEX_NIC_IFNAME:-eth0}                 # NCCL / GLOO socket interface
# export DEX_NODES=4                                            # optional: use the first N hosts of the hostfile (default: all)
# Software
export DEX_PYTHON=${DEX_PYTHON:-/path/to/env/bin/python}      # Python interpreter of the workload being run (requirements/)
export DEX_NVSHMEM_HOME=${DEX_NVSHMEM_HOME:-/path/to/nvshmem} # NVSHMEM install root (include/, lib/)
export DEX_CUDA_HOME=${DEX_CUDA_HOME:-/usr/local/cuda}        # CUDA toolkit matching the torch build
# export DEX_CUDA_ARCH=sm_86                                    # optional: nvcc target architecture of the extension
# export DEX_EXT_DIR=/path/to/ext                               # optional: extension built with build_ext.sh --out (W4 scripts default to build/ext_bf16)
# export DEX_INFLLM_DTYPE=fp16                                  # set by build_ext.sh --infllm-dtype: fp16, or bf16 for W4's kernels
# Data and results
export DEX_DATA_ROOT=${DEX_DATA_ROOT:-/path/to/data}          # masks, dumps, selections, plans
export DEX_MODEL_ROOT=${DEX_MODEL_ROOT:-/path/to/models}      # model checkpoints
export DEX_RESULT_ROOT=${DEX_RESULT_ROOT:-./results}          # where experiments write jsonl and logs
export DEX_SCRATCH=${DEX_SCRATCH:-/tmp}                       # node-local scratch
# Optional, baselines only
# export DEX_GUROBI_LICENSE=/path/to/gurobi.lic                 # UltraAttn's deployment ILP
# export DEX_FA283_PATH=/path/to/flash-attention-2.8.3          # official FlashAttention 2.8.3 build for the UltraAttn executor
# export DEX_DCP_PYTHON=/path/to/dcp_env/bin/python             # Python environment with the DCP package
# export DEX_MAGI_PYTHON=/path/to/magi_env/bin/python           # Python environment with MagiAttention
# export DEX_REDIS_SERVER=/path/to/redis-server                 # DCP's data loader: the redis-server binary
# export DEX_MAGI_CLEAR_CACHE=1                                 # Magi training: clear PyTorch's allocator cache after every attention call (0 keeps it)
# export DEX_COMM_GPU=0                                         # GPU used by the communication-volume planners
# export DEX_CALIBRATION=/path/to/calibration.json              # cost-model constants, default config/calibration.json
# Diagnostics
# export DEX_TRACE=1                                            # per-worker time counters of the kernels (not during timed runs)
