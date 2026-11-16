"""Machine-specific configuration lookup. Every path, host and interface name comes from the environment that
config/env.sh exports (template: config/env.example.sh); cfg() fails with a message naming the template when a variable
is unset, so no site-specific default is ever used silently."""
import os

TEMPLATE = "config/env.example.sh"


def cfg(name: str, default: str | None = None) -> str:
    """the value of environment variable `name`; `default` only for variables documented as optional"""
    v = os.environ.get(name)
    if v is not None and v != "":
        return v
    if default is not None:
        return default
    raise RuntimeError(f"environment variable {name} is not set: copy {TEMPLATE} to config/env.sh, fill it in and "
                       f"`source config/env.sh` before running")


def repo_root() -> str:
    """the release checkout (two levels above this file)"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
