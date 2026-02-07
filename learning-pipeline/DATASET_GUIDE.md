# 📚 Dataset Integration Guide for Hive-Mind

**Purpose:** Boost your model's capabilities by adding high-quality training data

**Current state:** 21 examples (synthetic tool interactions)
**Target:** 1,000-10,000 examples for production quality
**Recommended:** Start with 5,000 curated examples

---

## 🎯 Recommended Strategy for YOUR Use Case

### Phase 1: Foundation (Start Here) ✅

**Priority Datasets:**
1. **Code Alpaca** (5,000 examples) - Code understanding
2. **Glaive Tools** (5,000 examples) - Tool/function calling
3. **Keep your real data** (21 examples) - Your actual usage patterns

**Why this combo:**
- Code Alpaca: Teaches code generation and explanation
- Glaive: Teaches tool selection and usage
- Real data: Keeps it grounded in YOUR actual use case

**Total:** ~10,000 examples (optimal for LoRA fine-tuning)

### Phase 2: Specialization (Later)

Add domain-specific data as needed:
- Bash commands dataset if you do lots of shell work
- SQL queries if you interact with databases
- DevOps runbooks if you do infrastructure work

---

## 🚀 Quick Start

### Download Recommended Datasets

```bash
cd /var/mnt/build/MCP/hive-mind/learning-pipeline

# Download all recommended datasets (5K each)
python3 scripts/download_datasets.py \\
    --output data/external \\
    --max-per-dataset 5000 \\
    --datasets all
```

This will create:
- `data/external/code_alpaca.jsonl` (~5K examples)
- `data/external/glaive_tools.jsonl` (~5K examples)
- `data/external/bash_commands.jsonl` (varies)
- `data/external/merged_dataset.jsonl` (all combined)

### Merge with Your Real Data

```bash
# Combine external data with your actual usage data
cat data/external/merged_dataset.jsonl \\
    data/automated/training_data_*.jsonl \\
    > data/training_dataset_full.jsonl

# Check the result
wc -l data/training_dataset_full.jsonl
```

### Train on Combined Dataset

```bash
# One-time training on full dataset
python3 scripts/train_lora.py \\
    --model "Qwen/Qwen2.5-0.5B" \\
    --dataset data/training_dataset_full.jsonl \\
    --output models/foundation_model \\
    --epochs 3 \\
    --batch-size 4 \\
    --grad-accum 4 \\
    --lora-r 16 \\
    --lora-alpha 32
```

**Note:** Larger dataset = longer training (est. 45-60 minutes for 10K examples)

---

## 📊 Dataset Comparison

| Dataset | Size | Quality | Relevance | Download Time | Training Time |
|---------|------|---------|-----------|---------------|---------------|
| **Code Alpaca** | 20K | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~2 min | ~45 min (5K) |
| **Glaive Tools** | 110K | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~5 min | ~45 min (5K) |
| **NL2Bash** | 9K | ⭐⭐⭐ | ⭐⭐⭐⭐ | ~1 min | ~15 min (all) |
| **Your Real Data** | 21 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | N/A | <1 min |

**Quality:** Data cleanliness and formatting
**Relevance:** How well it matches your MCP tool-use case

---

## 🎨 Custom Dataset Creation

Want to add your own data? Here's the format:

```json
{
  "user_request": "What the user asked for",
  "tool": "tool_name",
  "command": "actual command or input",
  "output": "what the tool returned",
  "timestamp": "2026-02-06T12:00:00Z",
  "metadata": {
    "source": "custom",
    "quality": "high"
  }
}
```

**Example custom datasets:**
1. Your command history (`~/.bash_history` → training data)
2. Git commit messages + diffs
3. API interaction logs
4. DevOps runbook procedures
5. Troubleshooting sessions

---

## ⚖️ Data Mixing Strategy

### Recommended Mix for Hive-Mind

```
Foundation Model (10,000 examples):
├─ 40% Code Alpaca (4,000)    - Code understanding
├─ 40% Glaive Tools (4,000)   - Tool usage
├─ 15% Bash Commands (1,500)  - Shell operations
└─ 5% Your Real Data (500)    - Actual usage patterns
```

**Why this ratio:**
- Large public datasets provide general capabilities
- Small real data keeps it grounded in YOUR use case
- Prevents overfitting to synthetic data

### Continuous Learning Mix

After foundation model, daily training should be:
```
Daily Updates (ongoing):
├─ 95% Your Real Data (from Redis queue)
└─ 5% Random samples from foundation datasets (prevent catastrophic forgetting)
```

---

## 🔬 Data Quality Tips

### What Makes Good Training Data

✅ **Good:**
- Clear input → output mapping
- Diverse tool usage
- Real-world examples
- Consistent formatting
- Error examples (showing how to recover)

❌ **Bad:**
- Duplicate examples (wastes training time)
- Inconsistent format
- Too generic ("Hello world" examples)
- No context
- Only success cases (model won't learn error handling)

### Filtering Bad Data

```bash
# Remove duplicates
sort -u data/merged_dataset.jsonl > data/deduped.jsonl

# Filter by length (too short = probably junk)
python3 << 'EOF'
import json
with open('data/deduped.jsonl') as f:
    lines = [line for line in f if len(json.loads(line)['output']) > 10]
with open('data/filtered.jsonl', 'w') as f:
    f.writelines(lines)
EOF
```

---

## 📈 Expected Improvements

### With Foundation Dataset (10K examples)

**Baseline (21 examples):**
- Loss: ~4.75
- Can handle: 20 tool types
- Quality: Basic

**After Foundation (10K examples):**
- Loss: ~2.5-3.0 (expected)
- Can handle: 100+ tool types
- Quality: Good general tool use
- New capabilities: Code explanation, better error handling

**After 30 days continuous learning (10K + 600 real):**
- Loss: ~2.0-2.5 (expected)
- Can handle: Your specific workflow patterns
- Quality: Excellent for YOUR use case

---

## 🎯 My Specific Recommendations for YOU

Based on your Hive-Mind setup:

### Start Here (This Weekend):

```bash
# 1. Download foundation datasets
python3 scripts/download_datasets.py --max-per-dataset 5000

# 2. Train foundation model
python3 scripts/train_lora.py \\
    --model "Qwen/Qwen2.5-0.5B" \\
    --dataset data/external/merged_dataset.jsonl \\
    --output models/foundation_v1 \\
    --epochs 3 \\
    --batch-size 4 \\
    --grad-accum 4 \\
    --lora-r 16 \\
    --lora-alpha 32

# 3. Test it
# Use the foundation model as your base for daily training
```

### Ongoing (Automated):

```bash
# Keep daily training running on Redis data
# It will naturally blend foundation knowledge with your usage
```

### Advanced (Optional):

1. **Collect your bash history:**
   ```bash
   # Create dataset from your command history
   python3 scripts/create_dataset_from_history.py
   ```

2. **Add domain-specific data:**
   - If you use git a lot: Add git workflow examples
   - If you debug often: Add debugging session logs
   - If you deploy: Add deployment runbooks

---

## 💾 Storage Considerations

**Datasets on disk:**
- Code Alpaca (5K): ~5 MB
- Glaive Tools (5K): ~8 MB
- Merged (10K): ~15 MB

**Models after training:**
- Each LoRA adapter: ~17 MB
- Foundation model adapter: ~17 MB (same size!)

**Total additional storage:** ~50 MB (negligible)

---

## 🚨 Common Pitfalls

### ❌ Don't Do This:

1. **Training on 100K examples**
   - LoRA doesn't need that much
   - 5-10K is optimal
   - More = longer training, not better results

2. **Only using public datasets**
   - Model won't know YOUR specific tools
   - Keep collecting real data!

3. **Mixing incompatible formats**
   - All data must follow the same schema
   - Use the conversion script

4. **Ignoring data quality**
   - 1,000 good examples > 10,000 mediocre ones
   - Quality > Quantity

### ✅ Do This Instead:

1. **Curate 5-10K best examples**
2. **Mix 90% public + 10% real**
3. **Validate format consistency**
4. **Filter for quality**

---

## 📋 Checklist

Before training on new datasets:

- [ ] Downloaded datasets to `data/external/`
- [ ] Checked format matches your schema
- [ ] Removed duplicates
- [ ] Merged with real data
- [ ] Backed up current model
- [ ] Allocated time for training (est. 45-60 min)
- [ ] Have monitoring ready (TensorBoard or logs)

---

## 🎓 Next Steps

**Immediate (This Week):**
1. Run `python3 scripts/download_datasets.py`
2. Train foundation model on 5-10K examples
3. Compare with current model (loss should drop significantly)

**Short-term (This Month):**
1. Let automated training continue with Redis data
2. Monitor loss trends (should decrease over time)
3. Test model quality on real tasks

**Long-term (This Year):**
1. Collect 10K+ real usage examples
2. Periodically retrain foundation model
3. Consider model versioning (v1.0, v2.0, etc.)

---

**Ready to boost your model?** Start with the download script and let me know if you hit any issues!
