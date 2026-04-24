#!/bin/bash
# Benchmark sweep for the writeup.
#
# Measures two things:
#   (A) single-GPU throughput vs grid size  -> bench_size.csv
#   (B) multi-GPU embarrassingly parallel sweep wall-clock
#       at 1, 2, 4 GPUs, all running the same size -> bench_gpu.csv
#
# Frames are suppressed (frame_stride > nsteps) so I/O doesn't pollute the
# per-step timing.
#
# Submit on Frontera:
#   cd $WORK/neural-field-gpu
#   sbatch scripts/benchmark.sh
#
# After it finishes, send me bench_size.csv and bench_gpu.csv (or the full
# log) and I'll regenerate the plots + recompile the PDF.

#SBATCH -J nf_bench
#SBATCH -o logs/nf_bench.%j.out
#SBATCH -e logs/nf_bench.%j.err
#SBATCH -p rtx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ASC25001

set -o pipefail
mkdir -p logs bench_out

module load cuda

echo "=========================================="
echo "neural field benchmark sweep"
echo "date : $(date)  host : $(hostname)"
echo "=========================================="

rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > bench_out/cmake.log 2>&1
cmake --build build -j >> bench_out/cmake.log 2>&1
if [[ ! -x build/neural_field ]]; then
  echo "build failed"; cat bench_out/cmake.log; exit 1
fi
BIN=$(pwd)/build/neural_field

# ---------------------------------------------------------------------
# (A) grid-size sweep on GPU 0
# ---------------------------------------------------------------------

CSV_A=bench_size.csv
echo "Nx,Ny,nsteps,total_s,per_step_ms,throughput_Gcell_s" > "${CSV_A}"

declare -a SIZES=(128 256 512 1024 2048 4096)
declare -a STEPS=(20000 10000 8000 4000 1500 400)

for i in 0 1 2 3 4 5; do
  N=${SIZES[$i]}
  S=${STEPS[$i]}
  STRIDE=$((S + 1))
  echo
  echo "--- grid ${N}x${N}, ${S} steps ---"
  CUDA_VISIBLE_DEVICES=0 "${BIN}" \
      --preset=spirals --nx=${N} --ny=${N} \
      --nsteps=${S} --frame-stride=${STRIDE} \
      --out=bench_out/discard 2>&1 | tee bench_out/size_${N}.log \
      | grep -E '^(total|per step|throughput|CSV,)' || true
  grep '^CSV,' bench_out/size_${N}.log | sed 's/^CSV,//' >> "${CSV_A}"
  rm -rf bench_out/discard
done

# ---------------------------------------------------------------------
# (B) multi-GPU concurrent sweep -- launch k copies in parallel and time
# the slowest.  k = 1, 2, 4.  Each at 1024x1024 / 4000 steps.
# ---------------------------------------------------------------------

CSV_B=bench_gpu.csv
echo "ngpus,wall_s,per_process_avg_s" > "${CSV_B}"

run_parallel() {
  local K=$1
  local T0=$(date +%s.%N)
  local PIDS=()
  local LOGS=()
  for g in $(seq 0 $((K-1))); do
    local LG=bench_out/gpu${K}_${g}.log
    LOGS+=("$LG")
    CUDA_VISIBLE_DEVICES=$g "${BIN}" \
        --preset=spirals --nx=1024 --ny=1024 \
        --nsteps=4000 --frame-stride=99999 \
        --out=bench_out/discard_$g > "$LG" 2>&1 &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait $p; done
  local T1=$(date +%s.%N)
  local WALL=$(awk -v a="$T0" -v b="$T1" 'BEGIN { printf "%.6f", b-a }')

  # average per-process total time from CSV lines
  local SUM=0.0
  for lg in "${LOGS[@]}"; do
    local T=$(grep '^CSV,' "$lg" | awk -F, '{ print $4 }')
    SUM=$(awk -v s="$SUM" -v t="$T" 'BEGIN { printf "%.6f", s+t }')
  done
  local AVG=$(awk -v s="$SUM" -v k="$K" 'BEGIN { printf "%.6f", s/k }')
  echo "${K},${WALL},${AVG}" >> "${CSV_B}"
  rm -rf bench_out/discard_*
}

echo
echo "=========================================="
echo "(B) multi-GPU parallel sweep (1024x1024)"
echo "=========================================="
for K in 1 2 4; do
  echo "--- ${K} concurrent GPU(s) ---"
  run_parallel $K
done

echo
echo "=========================================="
echo "size sweep    -> ${CSV_A}"
cat "${CSV_A}"
echo
echo "multi-GPU sweep -> ${CSV_B}"
cat "${CSV_B}"
echo "=========================================="
