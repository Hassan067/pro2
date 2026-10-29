#!/usr/bin/env bash
# Final render pipeline: Cycles aircraft layer and the cloud/plume compositor run
# concurrently (the compositor waits for each Cycles frame).  Both skip finished
# frames, so the script can simply be re-run to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/user/bvenv/bin/python}
A=${1:-1}; B=${2:-420}
mkdir -p work/cyc work/comp work/logs
$PY scripts/build_scene.py > work/logs/build.log 2>&1
$PY scripts/export_frames.py --blend blend/ep01_afterburners.blend --out work/frame_data.json > work/logs/export.log 2>&1
$PY scripts/render.py --blend blend/ep01_afterburners.blend --out work/cyc --range "$A" "$B" --skip-existing \
    > work/logs/cycles.log 2>&1 &
CYC=$!
$PY scripts/compose.py --data work/frame_data.json --cycles work/cyc --out work/comp --range "$A" "$B" \
    --cloud-scale 0.5 --samples 3 --plume-scale 2 --skip-existing --wait > work/logs/compose.log 2>&1
wait $CYC
echo "ALL DONE"
