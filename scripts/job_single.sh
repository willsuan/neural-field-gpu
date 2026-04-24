#!/bin/bash
# Single-GPU Wilson-Cowan run on a Frontera RTX node.
# Usage: sbatch scripts/job_single.sh [preset]
#   preset defaults to "spirals"

#SBATCH -J nf_single
#SBATCH -o logs/nf_single.%j.out
#SBATCH -e logs/nf_single.%j.err
#SBATCH -p rtx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ASC25001

set -euo pipefail

PRESET="${1:-spirals}"
NX="${NX:-1024}"
NY="${NY:-1024}"
NSTEPS="${NSTEPS:-6000}"
FRAME_STRIDE="${FRAME_STRIDE:-20}"

module load cuda
mkdir -p logs

# build into ./build (fresh is safest for a job)
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

OUT="frames/${PRESET}"
mkdir -p "${OUT}"

./build/neural_field \
    --preset="${PRESET}" \
    --nx="${NX}" --ny="${NY}" \
    --nsteps="${NSTEPS}" \
    --frame-stride="${FRAME_STRIDE}" \
    --out="${OUT}"

echo
echo "frames in ${OUT}/"
echo "stitch to video with: scripts/make_video.sh ${OUT}"
