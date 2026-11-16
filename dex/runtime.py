"""NVSHMEM runtime of the Dex kernels: bootstrap from a torch.distributed process group. The symmetric heap is reserved
at initialisation (NVSHMEM_SYMMETRIC_SIZE, config/runtime_defaults.sh) and cannot grow afterwards; the kernels' symmetric
regions are allocated inside it by the extension."""
import torch.distributed as dist

from dex import dex_cuda  # compiled extension (build/build_ext.sh)


def init_nvshmem(pg: dist.ProcessGroup = None) -> None:
    """collective: bootstrap NVSHMEM over an existing process group (default: the world group)"""
    dex_cuda.init_nvshmem(pg or dist.group.WORLD)


def finalize_nvshmem() -> None:
    dex_cuda.finalize_nvshmem()
