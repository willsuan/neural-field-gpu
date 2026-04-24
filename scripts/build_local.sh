#!/bin/bash
# Convenience: configure + build locally (no SLURM).
# Useful for interactive testing on a login node in an `idev` session.

set -euo pipefail

if [[ -n "${TACC_SYSTEM:-}" ]]; then
  module load cuda
fi

rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo
echo "built ./build/neural_field"
echo "try: ./build/neural_field --preset=spirals --nsteps=2000 --out=frames/test"
