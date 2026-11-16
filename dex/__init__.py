"""DexAttention: distributed sparse attention through rapid online planning and persistent gpu execution.

Subpackages: planner (assigner), scheduler (schedule + kernel tables), attn (per-workload front-ends), common (layouts,
padding correction, FP64 check, configuration). `dex.runtime` wraps NVSHMEM start-up. The compiled extension is built
in place as `dex.dex_cuda` by build/build_ext.sh and always imported as `from dex import dex_cuda`, so the
process-wide NVSHMEM state of the extension is never loaded twice. DEX_EXT_DIR selects an extension built elsewhere
(build_ext.sh --out), for a workload whose Python environment or data type differs: its directory comes first on the
package path."""
import os

if os.environ.get("DEX_EXT_DIR"):
    __path__.insert(0, os.path.abspath(os.environ["DEX_EXT_DIR"]))
