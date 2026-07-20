#!/usr/bin/env bash
set -euo pipefail

mkdir -p data
checkpoint="data/groundingdino_swint_ogc.pth"
minimum_bytes=1000000
current_bytes=0
if [[ -f "${checkpoint}" ]]; then
  current_bytes=$(stat -c '%s' "${checkpoint}")
fi
if (( current_bytes >= minimum_bytes )); then
  echo "GroundingDINO checkpoint already exists; skipping download."
  exit 0
fi

temporary_checkpoint="${checkpoint}.partial"
curl -fL --retry 3 \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
  -o "${temporary_checkpoint}"
downloaded_bytes=$(stat -c '%s' "${temporary_checkpoint}")
if (( downloaded_bytes < minimum_bytes )); then
  echo "Downloaded GroundingDINO checkpoint is unexpectedly small: ${downloaded_bytes} bytes" >&2
  exit 1
fi
mv "${temporary_checkpoint}" "${checkpoint}"
echo "Downloaded ${checkpoint} (${downloaded_bytes} bytes)."
