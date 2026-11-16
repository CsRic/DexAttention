#!/bin/bash
# Builds the scheduler's C++ libraries into dex/scheduler/lib/.
# Inputs : csrc/fastsched_blocks.cpp (forward schedule + tables), csrc/bwdplan.cpp (backward; includes the former).
# Outputs: lib/libfsched.so, lib/libbwdplan.so. Example: bash dex/scheduler/build.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p lib
g++ -O3 -march=native -std=c++17 -shared -fPIC -o lib/libfsched.so.tmp csrc/fastsched_blocks.cpp
g++ -O3 -march=native -std=c++17 -shared -fPIC -pthread -o lib/libbwdplan.so.tmp csrc/bwdplan.cpp
mv lib/libfsched.so.tmp lib/libfsched.so
mv lib/libbwdplan.so.tmp lib/libbwdplan.so
echo "built lib/libfsched.so lib/libbwdplan.so"
