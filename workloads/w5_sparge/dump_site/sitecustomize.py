# Put this directory on PYTHONPATH to install the W5 q / k dump in every Python process of a generation run, including
# SGLang Diffusion's spawned workers (active only when W5_DUMP_DIR is set).
import os
import sys

if os.environ.get("W5_DUMP_DIR"):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import dump_hook

        dump_hook.install()
    except ImportError:  # processes without spas_sage_attn (launcher shells)
        pass
