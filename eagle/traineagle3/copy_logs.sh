#!/bin/bash
# Copy TensorBoard log files to local log directory
# Usage: bash copy_logs.sh <checkpoint_dir> [local_folder_name]
#
# Example:
#   bash copy_logs.sh ./checkpoints_qat_2gpu lr5e5-5ep-2gpu-1layer-littlebit
#   bash copy_logs.sh ./checkpoints_qat_03   lr5e5-5ep-4gpu-1layer-littlebit-03

LOCAL_LOG_BASE="/c/log/speculative_decoding"
# If running from WSL or Git Bash, use Windows-style path:
# LOCAL_LOG_BASE="C:/log/speculative_decoding"

CHECKPOINT_DIR="${1:?Usage: $0 <checkpoint_dir> [local_folder_name]}"
FOLDER_NAME="${2:-$(basename "$CHECKPOINT_DIR")}"

SRC_RUNS_DIR="${CHECKPOINT_DIR}/runs"
DST_DIR="${LOCAL_LOG_BASE}/${FOLDER_NAME}"

if [ ! -d "$SRC_RUNS_DIR" ]; then
    echo "[ERROR] Source runs directory not found: $SRC_RUNS_DIR"
    exit 1
fi

echo "[INFO] Creating destination: $DST_DIR"
mkdir -p "$DST_DIR"

echo "[INFO] Copying TensorBoard logs..."
cp -v "$SRC_RUNS_DIR"/events.out.tfevents.* "$DST_DIR/"

echo "[DONE] Logs copied to: $DST_DIR"
echo "[TIP]  Run: tensorboard --logdir $LOCAL_LOG_BASE"
