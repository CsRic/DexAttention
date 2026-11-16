"""DCP baseline: the official planner and executor (third_party/dcp, third_party/PINS.md), driven per input by
track.py (one persistent launch per list of inputs), ladder.py (block-size ladder, measurement launches), e2e.py (training).
Importing the package configures DCP for the evaluation, before DCP is imported: FP32 partial results and the comm / compute
cost-model constants of the target cluster (Gbps, ms, TFLOP/s, GB/s), measured on that cluster; override per cluster through
the environment."""
import os
import resource
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_ROOT, "third_party", "dcp") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "third_party", "dcp"))
_FA = os.environ.get("DEX_FA283_PATH")   # DCP's microbenchmark module imports flash_attn when it loads
if _FA and _FA not in sys.path:
    sys.path.insert(0, _FA)
DCP_ENV = dict(DCP_FP32_OUT="1", DCP_INTER_GBPS="14.69", DCP_INTRA_GBPS="124.6", DCP_INTER_LAT_MS="0.055",
               DCP_INTRA_LAT_MS="0.055", DCP_PEAK_TFLOPS="111", DCP_PEAK_MEMBW="768", OMP_NUM_THREADS="1")
for _k, _v in DCP_ENV.items():
    os.environ.setdefault(_k, _v)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(1048576, resource.getrlimit(resource.RLIMIT_NOFILE)[1]),) * 2)  # DCP's dataloader passes tensors as fds
import torch  # noqa: E402,F401  (before numpy: DCP's PyTorch build needs a more recent C++ runtime than the one numpy would load)
