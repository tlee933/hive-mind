# 🤖 Automated Training Setup

**Status:** ✅ Fully Configured and Running
**Schedule:** Daily at 2:00 AM MST
**Next Run:** Check with `systemctl list-timers hive-mind-training.timer`

---

## Overview

The Hive-Mind learning pipeline now runs automatically every day at 2:00 AM, collecting data from Redis and training new LoRA adapters.

### What Happens Automatically

```
┌──────────────────────────────────────────────────────┐
│  DAILY AUTOMATED WORKFLOW (2:00 AM)                  │
├──────────────────────────────────────────────────────┤
│  1. Check Redis connection                           │
│  2. Collect training data from learning queue        │
│  3. Check if enough examples (min: 10)               │
│  4. Train LoRA model (3 epochs)                      │
│  5. Save model with timestamp                        │
│  6. Update 'latest' symlink                          │
│  7. Generate training report                         │
│  8. Clean up old files (>30 days)                    │
└──────────────────────────────────────────────────────┘
```

---

## 📊 Monitoring

### Check Timer Status

```bash
# View timer status
systemctl status hive-mind-training.timer

# List when next run is scheduled
systemctl list-timers hive-mind-training.timer

# View timer logs
journalctl -u hive-mind-training.timer -f
```

### Check Training Logs

```bash
# View latest training log
ls -lt logs/training_*.log | head -1 | awk '{print $NF}' | xargs tail -f

# View all recent runs
ls -lt logs/training_*.log | head -5

# View systemd logs
tail -f logs/systemd.log
```

### Check Reports

```bash
# View latest training report
ls -lt logs/report_*.txt | head -1 | awk '{print $NF}' | xargs cat

# View all recent reports
ls -lt logs/report_*.txt | head -5
```

---

## 🎯 Current Configuration

**Training Parameters:**
- Base model: `Qwen/Qwen2.5-0.5B`
- LoRA rank: 8
- LoRA alpha: 16
- Epochs: 3
- Batch size: 2
- Gradient accumulation: 4
- Learning rate: 2e-4

**Minimum Requirements:**
- At least 10 examples in Redis queue
- Redis cluster accessible
- 12GB VRAM available

**Storage:**
- Models: `models/automated/`
- Data: `data/automated/`
- Logs: `logs/`

**Retention Policy:**
- Files older than 30 days are automatically deleted

---

## 🔧 Management Commands

### Manual Trigger

Run training immediately (doesn't wait for scheduled time):

```bash
# Trigger training now
sudo systemctl start hive-mind-training.service

# Watch it run
journalctl -u hive-mind-training.service -f
```

### Disable/Enable

```bash
# Temporarily stop automated training
sudo systemctl stop hive-mind-training.timer

# Permanently disable
sudo systemctl disable hive-mind-training.timer

# Re-enable
sudo systemctl enable hive-mind-training.timer
sudo systemctl start hive-mind-training.timer
```

### Modify Schedule

Edit the timer file:

```bash
sudo systemctl edit --full hive-mind-training.timer
```

Change the `OnCalendar` line:
- `*-*-* 02:00:00` = Daily at 2 AM
- `Mon *-*-* 02:00:00` = Monday at 2 AM
- `*-*-01 02:00:00` = 1st of every month at 2 AM

Then reload:

```bash
sudo systemctl daemon-reload
sudo systemctl restart hive-mind-training.timer
```

---

## 📁 File Locations

### Scripts
- Main script: `scripts/automated_training.sh`
- Data collector: `scripts/collect_data.py`
- Training script: `scripts/train_lora.py`

### Systemd Units
- Service: `/etc/systemd/system/hive-mind-training.service`
- Timer: `/etc/systemd/system/hive-mind-training.timer`

### Outputs
- Models: `models/automated/model_YYYYMMDD_HHMMSS/`
- Latest symlink: `models/automated/latest` → latest trained model
- Training data: `data/automated/training_data_YYYYMMDD_HHMMSS.jsonl`
- Logs: `logs/training_YYYYMMDD_HHMMSS.log`
- Reports: `logs/report_YYYYMMDD_HHMMSS.txt`

---

## 🚨 Troubleshooting

### Training Didn't Run

```bash
# Check timer status
systemctl status hive-mind-training.timer

# Check if service failed
systemctl status hive-mind-training.service

# View error logs
journalctl -u hive-mind-training.service --since "24 hours ago"
```

### Not Enough Training Data

If you see "Not enough examples for training":
- Check Redis queue: needs at least 10 examples
- Add more data to Redis learning queue
- Lower `MIN_EXAMPLES` in `scripts/automated_training.sh`

### Training Failed

```bash
# Check the error log
tail -100 logs/systemd-error.log

# Common issues:
# - Redis connection failed → Check Redis cluster status
# - CUDA/HIP error → Check GPU availability with rocm-smi
# - Permission denied → Check file permissions
```

---

## 📈 Performance Metrics

### Expected Training Times

Based on queue size (with current config):

| Examples | Training Time | Model Size |
|----------|--------------|------------|
| 10-20    | ~5 seconds   | 17 MB      |
| 50-100   | ~15 seconds  | 17 MB      |
| 500-1000 | ~2 minutes   | 17 MB      |
| 5000+    | ~15 minutes  | 17 MB      |

### Disk Usage

- Each model: ~17 MB
- Each dataset: ~100 KB per 100 examples
- Each log: ~50 KB
- Total daily: ~20 MB (with 100 examples)

Files are automatically cleaned up after 30 days.

---

## 🔄 Model Deployment

### Using the Latest Model

The `latest` symlink always points to the most recently trained model:

```bash
cd models/automated

# Check which model is latest
ls -l latest

# Use in inference
python3 ../../scripts/inference.py --model latest/
```

### Manual Model Selection

```bash
# List all trained models
ls -lt models/automated/model_*

# Use specific model
python3 scripts/inference.py --model models/automated/model_20260205_220000/
```

---

## 📊 Training Report Example

After each run, a report is generated:

```
===============================================================================
Hive-Mind Training Report
===============================================================================
Date: Thu Feb  5 10:20:31 PM MST 2026
Run ID: 20260205_222004

DATA COLLECTION
---------------
Dataset: /path/to/training_data_20260205_222004.jsonl
Examples collected: 21
Source: Redis learning queue

TRAINING CONFIGURATION
----------------------
Base model: Qwen/Qwen2.5-0.5B
LoRA rank: 8
LoRA alpha: 16
Epochs: 3
Batch size: 2
Gradient accumulation: 4
Learning rate: 2e-4

MODEL OUTPUT
------------
Location: /path/to/model_20260205_222004
Adapter size: 17M
Symlink: /path/to/latest

METRICS
-------
Training runtime: 12.07 seconds
Samples/second: 5.218
Steps/second: 0.745
Final loss: 4.753
===============================================================================
```

---

## ✅ Verification Checklist

To verify everything is working:

- [ ] Timer is active: `systemctl is-active hive-mind-training.timer`
- [ ] Timer is enabled: `systemctl is-enabled hive-mind-training.timer`
- [ ] Next run scheduled: `systemctl list-timers hive-mind-training.timer`
- [ ] Redis has data: Check queue length
- [ ] Logs directory writable: `ls -ld logs/`
- [ ] Models directory writable: `ls -ld models/automated/`

---

## 🎉 Success Indicators

Your automated training is working if you see:

1. **Timer shows next run**: `systemctl list-timers` shows future time
2. **New models appear daily**: `ls -lt models/automated/` shows recent directories
3. **Logs are created**: `ls -lt logs/training_*.log` shows daily logs
4. **Latest symlink updates**: `ls -l models/automated/latest` changes timestamp
5. **Reports generated**: `ls -lt logs/report_*.txt` shows new reports

---

**Setup Date:** February 5, 2026
**Setup Status:** ✅ Complete and Operational
**Next Steps:** Monitor for 7 days to ensure stability

For questions or issues, check the logs first!
