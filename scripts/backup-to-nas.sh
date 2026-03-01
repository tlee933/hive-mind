#!/bin/bash
# HiveMind Model Backup - rsync to NAS with rotation
# Timer: hivecoder-backup.timer (daily 3 AM)
# Keeps 3 rotating backups

set -e

NAS_PATH="/var/mnt/ai/hive-mind/model-archives"
SOURCE="/var/mnt/build/MCP/hive-mind/learning-pipeline/models"
LOG_DIR="/var/mnt/build/MCP/hive-mind/logs"
DATE=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/backup.log"

echo "[$DATE] Starting backup..." >> "$LOG"

# Create dated backup dir
BACKUP_DIR="${NAS_PATH}/backup-${DATE}"
mkdir -p "$BACKUP_DIR"

# Backup foundation models (7b and 14b if they exist)
for export_dir in foundation_7b_export foundation_14b_export; do
  if [ -d "$SOURCE/$export_dir" ]; then
    rsync -av \
      "$SOURCE/$export_dir/" "$BACKUP_DIR/$export_dir/" \
      >> "$LOG" 2>&1
    echo "[$DATE] Backed up $export_dir" >> "$LOG"
  fi
done

# Backup registry
rsync -av \
  "$SOURCE/registry/" "$BACKUP_DIR/registry/" \
  >> "$LOG" 2>&1

# Backup latest continuous model
DEPLOYED=$(ls -t "$SOURCE/continuous/" 2>/dev/null | head -1)
if [ -n "$DEPLOYED" ]; then
  mkdir -p "$BACKUP_DIR/continuous/$DEPLOYED"
  rsync -av \
    "$SOURCE/continuous/$DEPLOYED/" "$BACKUP_DIR/continuous/$DEPLOYED/" \
    >> "$LOG" 2>&1
  echo "[$DATE] Backed up continuous/$DEPLOYED" >> "$LOG"
fi

# Backup latest automated model
AUTOMATED=$(ls -t "$SOURCE/automated/" 2>/dev/null | grep -E '^model_' | head -1)
if [ -n "$AUTOMATED" ]; then
  mkdir -p "$BACKUP_DIR/automated/$AUTOMATED"
  rsync -av \
    "$SOURCE/automated/$AUTOMATED/" "$BACKUP_DIR/automated/$AUTOMATED/" \
    >> "$LOG" 2>&1
  echo "[$DATE] Backed up automated/$AUTOMATED" >> "$LOG"
fi

# Rotate backups - keep only 3 newest
cd "$NAS_PATH"
REMOVED=$(ls -dt backup-* 2>/dev/null | tail -n +4)
if [ -n "$REMOVED" ]; then
  echo "$REMOVED" | xargs rm -rf 2>/dev/null || true
  echo "[$DATE] Removed old backups: $(echo "$REMOVED" | tr '\n' ' ')" >> "$LOG"
fi

# Cleanup old local training logs (>30 days)
find "$LOG_DIR" -name "training_*.log" -mtime +30 -delete 2>/dev/null || true
find "$LOG_DIR" -name "report_*.txt" -mtime +30 -delete 2>/dev/null || true
find "$LOG_DIR" -name "backup.log" -size +10M -exec truncate -s 1M {} \; 2>/dev/null || true

# Cleanup old local training data (>30 days)
DATA_DIR="/var/mnt/build/MCP/hive-mind/learning-pipeline/data/automated"
find "$DATA_DIR" -name "*.jsonl" -mtime +30 -delete 2>/dev/null || true

# Cleanup old local model checkpoints (>30 days, keep latest symlink targets)
for model_dir in automated continuous; do
  LATEST=$(readlink -f "$SOURCE/$model_dir/latest" 2>/dev/null || true)
  find "$SOURCE/$model_dir" -maxdepth 1 -type d -name "model_*" -mtime +30 2>/dev/null | while read dir; do
    [ "$(readlink -f "$dir")" = "$LATEST" ] && continue
    rm -rf "$dir"
    echo "[$DATE] Cleaned old model: $dir" >> "$LOG"
  done
  find "$SOURCE/$model_dir" -maxdepth 1 -type d -name "v*" -mtime +30 2>/dev/null | while read dir; do
    [ "$(readlink -f "$dir")" = "$LATEST" ] && continue
    rm -rf "$dir"
    echo "[$DATE] Cleaned old model: $dir" >> "$LOG"
  done
done

KEPT=$(ls -d backup-* 2>/dev/null | wc -l)
echo "[$DATE] Backup complete: $BACKUP_DIR (kept $KEPT backups)" >> "$LOG"
