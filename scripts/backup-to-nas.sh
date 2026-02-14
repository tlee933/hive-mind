#!/bin/bash
# HiveMind Model Backup - rsync to NAS with rotation
# Cron: 0 3 * * 0,3,5 (Sun/Wed/Fri 3am)
# Keeps 3 rotating backups

set -e

NAS_PATH="/var/mnt/ai/hive-mind/model-archives"
SOURCE="/var/mnt/build/MCP/hive-mind/learning-pipeline/models"
DATE=$(date +%Y%m%d_%H%M%S)
LOG="/var/mnt/build/MCP/hive-mind/logs/backup.log"

echo "[$DATE] Starting backup..." >> "$LOG"

# Create dated backup dir
BACKUP_DIR="${NAS_PATH}/backup-${DATE}"
mkdir -p "$BACKUP_DIR"

# Rsync models (foundation + current deployed + registry)
rsync -av \
  "$SOURCE/foundation_7b_export/" "$BACKUP_DIR/foundation_7b_export/" \
  >> "$LOG" 2>&1

rsync -av \
  "$SOURCE/registry/" "$BACKUP_DIR/registry/" \
  >> "$LOG" 2>&1

# Get current deployed version from registry
DEPLOYED=$(ls -t "$SOURCE/continuous/" 2>/dev/null | head -1)
if [ -n "$DEPLOYED" ]; then
  rsync -av \
    "$SOURCE/continuous/$DEPLOYED/" "$BACKUP_DIR/continuous/$DEPLOYED/" \
    >> "$LOG" 2>&1
fi

# Rotate - keep only 3 newest backups
cd "$NAS_PATH"
ls -dt backup-* 2>/dev/null | tail -n +4 | xargs rm -rf 2>/dev/null || true

echo "[$DATE] Backup complete: $BACKUP_DIR" >> "$LOG"
echo "[$DATE] Kept backups: $(ls -d backup-* 2>/dev/null | wc -l)" >> "$LOG"
