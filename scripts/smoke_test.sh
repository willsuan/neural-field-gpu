#!/bin/bash
# Smoke test for the Wilson-Cowan neural field project.
#
# Submits a short job to rtx-dev that: builds, runs a tiny case, and dumps
# everything useful (module list, nvcc/cmake/nvidia-smi, build logs, run
# output, frame file sizes) to the SLURM stdout log so you can paste it
# back to Claude for feedback.
#
# Usage:
#   cd project
#   sbatch scripts/smoke_test.sh
#
# After it finishes (~2-5 min), share:
#   logs/nf_smoke.<jobid>.out   <- most of it lives here
#   logs/nf_smoke.<jobid>.err   <- any cuda/cufft runtime errors
#   (optional) smoke_frames/frame_00009.ppm   for eyeballing output

#SBATCH -J nf_smoke
#SBATCH -o logs/nf_smoke.%j.out
#SBATCH -e logs/nf_smoke.%j.err
#SBATCH -p rtx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00
#SBATCH -A ASC25001

set -o pipefail
mkdir -p logs smoke_frames

echo "========================================"
echo "Wilson-Cowan neural field - smoke test"
echo "========================================"
echo "date     : $(date)"
echo "hostname : $(hostname)"
echo "job id   : ${SLURM_JOB_ID:-n/a}"
echo "pwd      : $(pwd)"
echo

echo "--- loading modules ---"
module load cuda
module load cmake 2>/dev/null || echo "(no cmake module, using default)"
module list 2>&1
echo

echo "--- nvcc --version ---"
nvcc --version 2>&1 || echo "(nvcc not found)"
echo

echo "--- cmake --version ---"
cmake --version 2>&1 | head -3 || echo "(cmake not found)"
echo

echo "--- nvidia-smi ---"
nvidia-smi 2>&1 || echo "(nvidia-smi failed)"
echo

echo "========================================"
echo "Build"
echo "========================================"
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
BUILD_RC=$?
echo "cmake exit: ${BUILD_RC}"
if [[ ${BUILD_RC} -ne 0 ]]; then
  echo "CMAKE CONFIGURE FAILED"
  exit 1
fi
cmake --build build -j
BUILD_RC=$?
echo "make exit: ${BUILD_RC}"
if [[ ! -x build/neural_field ]]; then
  echo "BUILD FAILED - no binary at build/neural_field"
  exit 1
fi
echo

echo "========================================"
echo "Tiny run: 256x256, 500 steps, preset=spirals"
echo "========================================"
rm -rf smoke_frames && mkdir -p smoke_frames
./build/neural_field \
    --preset=spirals \
    --nx=256 --ny=256 \
    --nsteps=500 \
    --frame-stride=50 \
    --out=smoke_frames
RUN_RC=$?
echo
echo "run exit code: ${RUN_RC}"
echo

echo "--- frames written ---"
ls -la smoke_frames 2>&1 | head -20
echo

echo "--- sanity: first and last frame sizes (should be 256*256*3 + tiny PPM header = ~196626 B) ---"
for f in smoke_frames/frame_00000.ppm smoke_frames/frame_00009.ppm; do
  if [[ -f "${f}" ]]; then
    ls -l "${f}"
    head -c 32 "${f}" | od -c | head -2
  else
    echo "(missing ${f})"
  fi
done
echo

echo "========================================"
echo "Done.  To share feedback:"
echo "  cat logs/nf_smoke.${SLURM_JOB_ID}.out  # paste into chat"
echo "  cat logs/nf_smoke.${SLURM_JOB_ID}.err  # paste if non-empty"
echo "  scp \$USER@frontera.tacc.utexas.edu:\$(pwd)/smoke_frames/frame_00009.ppm ."
echo "========================================"
