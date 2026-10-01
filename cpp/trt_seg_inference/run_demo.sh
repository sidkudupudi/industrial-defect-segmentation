#!/usr/bin/env bash
# Runs the C++ TensorRT app once per image. The app itself is unchanged: it reads ./test_defect.png and writes
# ./final_output_1024.jpg, so each image is copied into a scratch folder next to the engine, the app runs, and the
# output image + console log are collected as <out_dir>/<case>.jpg and <out_dir>/<case>.log.
#
# Usage:
#   run_demo.sh <inference_binary> <best_1024.engine> <out_dir> <cases.csv> <mvtec_root>   # case names from the CSV
#   run_demo.sh <inference_binary> <best_1024.engine> <out_dir> <image> [<image> ...]       # case name = file name
# cases.csv columns: case,category,defect,image  (image relative to <mvtec_root>)
set -euo pipefail
BIN=$(realpath "$1"); ENGINE=$(realpath "$2"); OUT=$(realpath -m "$3"); shift 3
WORK=$(mktemp -d); mkdir -p "$OUT"
ln -s "$ENGINE" "$WORK/best_1024.engine"

run_case() {   # $1 = case name, $2 = image path
  rm -f "$WORK/test_defect.png"
  cp --no-preserve=mode "$2" "$WORK/test_defect.png"     # dataset files may be read-only
  (cd "$WORK" && "$BIN" > "$OUT/$1.log" 2>&1)
  mv "$WORK/final_output_1024.jpg" "$OUT/$1.jpg"
  echo "$1: $(grep -c 'Defect Detected' "$OUT/$1.log" || true) detection(s)"
}

if [[ "$1" == *.csv ]]; then
  ROOT=$(realpath "$2")
  while IFS=, read -r case category defect image; do
    run_case "$case" "$ROOT/$image"
  done < <(tail -n +2 "$1")
else
  for img in "$@"; do run_case "$(basename "${img%.*}")" "$img"; done
fi
rm -rf "$WORK"
