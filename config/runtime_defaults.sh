# Runtime environment profiles. tools/launch.sh sources this file, applies dex_env_common, and applies the profile named by
# DEX_ENV_PROFILE if set; run scripts may instead source it and call a profile themselves. A value already in the
# environment always wins. NVSHMEM_SYMMETRIC_SIZE is not set here: every driver or run script sizes the heap per experiment.
# Inputs: DEX_NVSHMEM_HOME, DEX_NIC_IFNAME. Example: source config/runtime_defaults.sh; dex_env_training

# every system
dex_env_common() {
  : "${NCCL_IB_DISABLE:=0}"; export NCCL_IB_DISABLE                  # inter-node traffic uses InfiniBand
  : "${NCCL_NET_GDR_LEVEL:=SYS}"; export NCCL_NET_GDR_LEVEL          # NCCL: GPUDirect RDMA across root complexes (the baselines' inter-node path)
  : "${NCCL_NET_GDR_READ:=1}"; export NCCL_NET_GDR_READ              # the sending side reads GPU memory directly as well
  : "${NCCL_DEBUG:=WARN}"; export NCCL_DEBUG                         # warnings only
  : "${NCCL_SOCKET_IFNAME:=$DEX_NIC_IFNAME}"; export NCCL_SOCKET_IFNAME  # NCCL bootstrap interface
  : "${GLOO_SOCKET_IFNAME:=$DEX_NIC_IFNAME}"; export GLOO_SOCKET_IFNAME  # gloo interface (per-rank planning exchanges)
  : "${OMP_NUM_THREADS:=1}"; export OMP_NUM_THREADS                  # one host thread per rank
  export LD_LIBRARY_PATH=$DEX_NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}  # the extension links libnvshmem_host
}

# Dex's host library resolved before anything else is imported
dex_env_preload() {
  export LD_PRELOAD=$DEX_NVSHMEM_HOME/lib/libnvshmem_host.so.3${LD_PRELOAD:+:$LD_PRELOAD}
}

# one hardware queue per blocking copy-engine stream, apart from NCCL's streams
dex_env_queues() {
  : "${CUDA_DEVICE_MAX_CONNECTIONS:=32}"; export CUDA_DEVICE_MAX_CONNECTIONS
}

# Dex operator tests of W4 (the run script adds CUDA_DEVICE_MAX_CONNECTIONS=32 for every system)
dex_env_operator() {
  : "${NVSHMEM_DISABLE_CUDA_VMM:=1}"; export NVSHMEM_DISABLE_CUDA_VMM  # the heap is one plain allocation the NIC peer-memory path can pin
}

# Dex training and prefill (W2-W6), and the W5 and W6 operator tests: only the memory Dex uses is reserved
dex_env_training() {
  dex_env_operator
  dex_env_queues
  : "${NVSHMEM_MAX_TEAMS:=8}"; export NVSHMEM_MAX_TEAMS              # Dex uses only the world team; each team reserves sync memory
  : "${NVSHMEM_DISABLE_NCCL:=1}"; export NVSHMEM_DISABLE_NCCL        # no NVSHMEM collectives beyond barriers, so no NCCL communicator of its own
  : "${NVSHMEM_CUMEM_GRANULARITY:=2097152}"; export NVSHMEM_CUMEM_GRANULARITY  # heap rounding of 2 MiB, so a reservation costs what it asks for
}

# Dex operator tests of W1-W3: the training settings, without the preload
dex_env_static() {
  dex_env_training
  : "${DEX_PRELOAD_NVSHMEM:=0}"; export DEX_PRELOAD_NVSHMEM
}
