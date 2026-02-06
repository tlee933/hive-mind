#!/bin/bash
##############################################################################
# Setup automated training cron job
##############################################################################

SCRIPT_PATH="/var/mnt/build/MCP/hive-mind/learning-pipeline/scripts/automated_training.sh"
CRON_TIME="0 2 * * *"  # Daily at 2 AM
CRON_JOB="$CRON_TIME $SCRIPT_PATH >> /var/mnt/build/MCP/hive-mind/learning-pipeline/logs/cron.log 2>&1"

echo "================================================================================"
echo "🕐 Setting up Automated Training Cron Job"
echo "================================================================================"
echo ""
echo "Schedule: Daily at 2:00 AM"
echo "Script: $SCRIPT_PATH"
echo "Log: /var/mnt/build/MCP/hive-mind/learning-pipeline/logs/cron.log"
echo ""

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -F "$SCRIPT_PATH" >/dev/null; then
    echo "⚠️  Cron job already exists. Removing old entry..."
    crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH" | crontab -
fi

# Add new cron job
echo "📝 Adding cron job..."
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

# Verify
echo ""
echo "✅ Cron job installed!"
echo ""
echo "Current crontab:"
echo "--------------------------------------------------------------------------------"
crontab -l | grep "$SCRIPT_PATH"
echo "--------------------------------------------------------------------------------"
echo ""
echo "📋 Management Commands:"
echo ""
echo "  View logs:"
echo "    tail -f /var/mnt/build/MCP/hive-mind/learning-pipeline/logs/cron.log"
echo ""
echo "  View all cron jobs:"
echo "    crontab -l"
echo ""
echo "  Edit cron jobs:"
echo "    crontab -e"
echo ""
echo "  Remove this job:"
echo "    crontab -l | grep -v automated_training.sh | crontab -"
echo ""
echo "  Test run now:"
echo "    $SCRIPT_PATH"
echo ""
echo "================================================================================"
