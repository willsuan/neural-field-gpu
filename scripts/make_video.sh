#!/bin/bash
# Stitch PPM frames in a directory into an mp4 via ffmpeg.
# Usage: scripts/make_video.sh frames/spirals [out.mp4] [fps]

set -euo pipefail

DIR="${1:-frames}"
OUT="${2:-${DIR}.mp4}"
FPS="${3:-30}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not in PATH.  On Frontera: module load ffmpeg"
  exit 1
fi

ffmpeg -y -framerate "${FPS}" -i "${DIR}/frame_%05d.ppm" \
       -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow \
       "${OUT}"

echo "wrote ${OUT}"
