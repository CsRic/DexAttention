#!/usr/bin/env bash
# Fetches third_party/ at the pinned commits (PINS.md), with the nested submodules each one needs, and applies the evaluation's
# changes from build/patches/. Inside a git checkout of this tree each pin is recorded as a gitlink, so it also works right
# after `git init`; a tree without git gets plain clones. Run this rather than a recursive submodule update: UltraAttn's
# nested third_party/kernel_profiler is not public, and only the nested repositories listed below are needed.
# build/build_ext.sh applies the changes of flash-attention and of InfLLM-v2's CUTLASS, so those are not repeated here.
# Inputs : network access, .gitmodules (URLs), build/patches/*.patch. Outputs: third_party/<name>/.
# Example: bash third_party/setup_submodules.sh                           everything
#          bash third_party/setup_submodules.sh flash-attention infllmv2_cuda_impl   only what the Dex extension compiles against
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TP=$ROOT/third_party
CHANGES=$ROOT/build/patches
GIT_TREE=0
[ "$(git -C "$ROOT" rev-parse --show-toplevel 2>/dev/null || true)" = "$ROOT" ] && GIT_TREE=1

# name | commit | change file ("-" for none) | nested submodules to initialise (comma separated, "-" for none)
PINS="
flash-attention|6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15|-|csrc/cutlass
infllmv2_cuda_impl|93cf2ec28e5a7acebe3f0bb7329b6c73a1be91f6|-|csrc/cutlass
yunchang|a5f8f600d435ced180a97c91c68e87ff3cfb8f89|-|-
ultraattn|b5aa10e18f996de59f5237c00ae9094cf911b9c7|ultraattn.patch|-
dcp|e1e19b8222349bb98d632016dc7544e26e6bcdac|dcp.patch|-
dcp_flash_attention|a15bcc7cac1a854a77677e90118fedfe48533a2e|dcp_flash_attention.patch|csrc/cutlass
magiattention|a3276e7bf06027f1bac540e50187cbbc798517f4|magiattention.patch|magi_attention/functional/flash-attention
flex_prefill|a24b27bf117fd6ad4ba34b0db82c598e7786fd59|-|-
SpargeAttn|ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a|spargeattn.patch|-
"

apply_changes() {   # repository, change file; a no-op if already applied
  if ! git -C "$1" apply --reverse --check "$CHANGES/$2" 2>/dev/null; then git -C "$1" apply "$CHANGES/$2"; fi
}

fetch() {   # name, commit
  local path=third_party/$1 url
  url=$(git config -f "$ROOT/.gitmodules" "submodule.$path.url")
  if [ "$GIT_TREE" = 1 ]; then
    git -C "$ROOT" ls-files --stage -- "$path" | grep -q '^160000' \
      || git -C "$ROOT" update-index --add --cacheinfo "160000,$2,$path"
    git -C "$ROOT" submodule update --init -- "$path"
  else
    [ -e "$TP/$1/.git" ] || git clone "$url" "$TP/$1"
    git -C "$TP/$1" checkout -q "$2"
  fi
}

for line in $PINS; do
  IFS='|' read -r name commit change nested <<< "$line"
  if [ $# -gt 0 ] && [[ " $* " != *" $name "* ]]; then continue; fi
  fetch "$name" "$commit"
  if [ "$nested" != - ]; then git -C "$TP/$name" submodule update --init -- ${nested//,/ }; fi
  if [ "$change" != - ]; then apply_changes "$TP/$name" "$change"; fi
  if [ "$name" = magiattention ]; then
    # its FA4 backend builds against the CUTLASS commit that flash-attention above also pins
    git -C "$TP/$name/magi_attention/functional/flash-attention" submodule update --init -- csrc/cutlass
    apply_changes "$TP/$name/magi_attention/functional/flash-attention" magiattention_flash_attention.patch
  fi
done
