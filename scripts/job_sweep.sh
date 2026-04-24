#!/bin/bash
# Parameter-sweep run: launch four presets concurrently, one per GPU on a
# single Frontera RTX node.  Uses CUDA_VISIBLE_DEVICES to pin each process
# to its own GPU, then waits for all four to finish.

#SBATCH -J nf_sweep
#SBATCH -o logs/nf_sweep.%j.out
#SBATCH -e logs/nf_sweep.%j.err
#SBATCH -p rtx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A ASC25001

set -euo pipefail

NX="${NX:-1024}"
NY="${NY:-1024}"
NSTEPS="${NSTEPS:-6000}"
FRAME_STRIDE="${FRAME_STRIDE:-20}"

module load cuda
mkdir -p logs frames

rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

BIN="$(pwd)/build/neural_field"
PRESETS=(spots stripes spirals waves)

for i in 0 1 2 3; do
  PRESET="${PRESETS[$i]}"
  OUT="frames/${PRESET}"
  mkdir -p "${OUT}"
  echo "launching preset=${PRESET} on GPU ${i} -> ${OUT}"
  CUDA_VISIBLE_DEVICES=${i} "${BIN}" \
      --preset="${PRESET}" \
      --nx="${NX}" --ny="${NY}" \
      --nsteps="${NSTEPS}" \
      --frame-stride="${FRAME_STRIDE}" \
      --out="${OUT}" \
      > "logs/nf_sweep_${PRESET}.log" 2>&1 &
done

wait

echo
echo "sweep complete.  frames in frames/{spots,stripes,spirals,waves}/"
echo "stitch each with: scripts/make_video.sh frames/<preset>"
