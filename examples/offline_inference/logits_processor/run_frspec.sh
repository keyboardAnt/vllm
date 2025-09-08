#!/usr/bin/env bash
set -euo pipefail

export FRSPEC_HOT_TOKEN_IDS=/home/projects/dharel/nadavt/repos/vllm/examples/offline_inference/logits_processor/hot_ids.pt

# Prints for visibility
echo "[run_frspec] FRSPEC_HOT_TOKEN_IDS=${FRSPEC_HOT_TOKEN_IDS}"
if [[ -f "${FRSPEC_HOT_TOKEN_IDS}" ]]; then
  echo "[run_frspec] Hot-ids file exists. Details:"
  ls -lh "${FRSPEC_HOT_TOKEN_IDS}"
else
  echo "[run_frspec] ERROR: Hot-ids file not found at ${FRSPEC_HOT_TOKEN_IDS}" >&2
  exit 1
fi

echo "[run_frspec] Running frspec.py..."
python /home/projects/dharel/nadavt/repos/vllm/examples/offline_inference/logits_processor/frspec.py


