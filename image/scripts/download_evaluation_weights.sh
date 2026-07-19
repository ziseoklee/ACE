#!/usr/bin/env bash
set -euo pipefail

mkdir -p data
if [ -s data/groundingdino_swint_ogc.pth ]; then
  echo "GroundingDINO checkpoint already exists; skipping download."
  exit 0
fi

curl -L \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/weights/groundingdino_swint_ogc.pth \
  -o data/groundingdino_swint_ogc.pth
