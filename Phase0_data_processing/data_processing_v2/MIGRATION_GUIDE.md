# Migration Guide: Using V2 with Integrated Data Quality Fixes

## 🎯 TL;DR

**NEW WORKFLOW (Recommended):**
```bash
# Step 1: Process data with fixes integrated
python example_usage_v2.py

# Step 2: Train baseline (automatically uses clean data)
python train_baseline_recommender.py
```

**That's it!** The fixes are applied during data preprocessing, so your training code doesn't need v2 anymore.

---

## 📋 What Changed?

### Option A: Integrated Approach (RECOMMENDED) ✅

**Data Quality Fixes Applied During Preprocessing**

```
example_usage_v2.py
├── Load raw data
├── Run quality checks (BEFORE)
├── 🔧 Apply fixes:
│   ├── Filter empty histories
│   ├── Filter no-click impressions
│   └── Create popularity scores
├── Run quality checks (AFTER)
├── Preprocess data
└── Save to ./processed_data_v2/
    ├── news_features_train.csv
    ├── diversity_stats.json
    ├── encoders.json
    ├── popularity_scores.json  ← NEW
    └── quality_fixes_applied.json  ← NEW

train_baseline_recommender.py
├── Load from ./processed_data_v2/  ← Automatically uses clean data
├── Train recommender
└── Evaluate
```

**Advantages:**
- ✅ Fixes applied once during preprocessing
- ✅ Original training code works without changes
- ✅ Clear separation of concerns
- ✅ Can reuse clean data for multiple experiments
- ✅ Better for your write-up (clear methodology)

### Option B: Training-Time Fixes (Alternative)

**Data Quality Fixes Applied During Training**

```
example_usage.py  ← Original, no changes
└── Save to ./processed_data/

train_baseline_recommender_v2.py  ← New version
├── Load from ./processed_data/
├── 🔧 Apply fixes during training
├── Train recommender
└── Evaluate
```

**Disadvantages:**
- ⚠️ Fixes applied every time you train
- ⚠️ More complex training code
- ⚠️ Have to remember to use v2 script

---

## 🚀 Recommended: Use Option A (Integrated)

### Step-by-Step Migration

#### 1. Run the New Preprocessing Script

```bash
python example_usage_v2.py
```

**What it does:**
```
[STEP 1-4] Load data and check quality (same as before)
[STEP 5] 🔧 Apply fixes:
  ✓ Filter 46,065 empty histories
  ✓ Filter impressions with no clicks  
  ✓ Create popularity scores for 101,527 articles
[STEP 6] Check quality again (should show fixes worked)
[STEP 7-11] Preprocess and save (enhanced with fixes)
[STEP 12] Create before/after comparison report
```

**Output:**
```
./processed_data_v2/
├── news_features_train.csv          (same as v1)
├── diversity_stats.json              (same as v1)
├── encoders.json                     (same as v1)
├── sample_train_interactions.csv     (same as v1)
├── popularity_scores.json            🆕 NEW
├── quality_fixes_applied.json        🆕 NEW
└── dataset_summary.json              (enhanced)

./
├── data_quality_report_prefixes.txt   🆕 BEFORE fixes
└── data_quality_report_postfixes.txt  🆕 AFTER fixes
```

#### 2. Train Using Original Script (Updated)

```bash
python train_baseline_recommender.py
```

**What changed:**
- Now looks for `./processed_data_v2/` first
- Falls back to `./processed_data/` if v2 not found
- Automatically loads popularity scores if available
- **No other changes needed!**

#### 3. Compare Before/After

Check the reports:
```bash
# BEFORE fixes
cat data_quality_report_prefixes.txt

# AFTER fixes
cat data_quality_report_postfixes.txt
```

---

## 📊 What You Get with V2

### 1. Clean Data

**Before:**
- 46,065 impressions with empty histories (2.06%)
- Some impressions with no clicks
- No popularity fallback

**After:**
- ✅ All users have history ≥ 1 item
- ✅ All impressions have at least one click
- ✅ Popularity scores available for cold start

### 2. Better Documentation

**New files created:**
```
quality_fixes_applied.json:
{
  "empty_histories_filtered": true,
  "no_click_impressions_filtered": true,
  "popularity_baseline_created": true,
  "min_history_length": 1
}

quality_fixes_comparison.json:
{
  "before_fixes": { ... },
  "after_fixes": { ... },
  "actions_taken": [ ... ]
}
```

### 3. Two Quality Reports

**Pre-fix report:**
- Shows original data issues
- Documents what needed fixing

**Post-fix report:**
- Shows fixes worked
- Confirms data is clean

---

## 🎓 For Your Capstone Write-Up

### Section: Data Preprocessing

```markdown
## Data Quality and Preprocessing

### Quality Assessment
Initial analysis revealed data quality issues typical of 
real-world recommendation datasets:

1. **Cold Start Problem**: 46,065 impressions (2.06%) had users 
   with empty click histories, preventing content-based profile 
   construction.

2. **Evaluation Validity**: Some impressions contained no positive 
   labels (no clicks), making them unsuitable for evaluation.

3. **Category Imbalance**: Severe imbalance (32,020:1 ratio) with 
   sports and news dominating 61.5% of content.

### Data Cleaning Strategy
To address these issues, we implemented a three-step cleaning process:

1. **History Filtering**: Removed impressions where users had fewer 
   than 1 prior interaction (min_history_length=1).
   
2. **Label Validation**: Filtered impressions with no positive labels 
   to ensure valid evaluation samples.
   
3. **Popularity Baseline**: Created normalized popularity scores 
   based on click counts to handle remaining cold start cases.

The cleaning process removed 2.06% of impressions while preserving 
97.94% of data for model training and evaluation.

### Category Imbalance Handling
The severe category imbalance was intentionally preserved as it:
- Reflects real-world news distribution patterns
- Demonstrates the echo chamber problem in pure accuracy optimization
- Provides motivation for diversity-aware re-ranking (Phase 2)
- Establishes a baseline for measuring diversity improvements
```

---

## 🔄 If You Already Ran example_usage.py

### Quick Fix

```bash
# Run v2 to get clean data
python example_usage_v2.py

# Your existing training code will automatically use it
python train_baseline_recommender.py
```

The updated `train_baseline_recommender.py` automatically:
1. Looks for `./processed_data_v2/` first
2. Falls back to `./processed_data/` if not found
3. Loads popularity scores if available
4. Warns you if using old data format

---

## ❓ FAQ

### Q: Do I need train_baseline_recommender_v2.py?

**A: No!** Option A (integrated approach) is better:
- Fixes applied once during preprocessing
- Original training code works fine
- Cleaner separation of concerns

The v2 training script was created as Option B if you wanted fixes at training time, but Option A is preferred.

### Q: What if I already have ./processed_data/?

**A: Run example_usage_v2.py** to create ./processed_data_v2/ with fixes. Then use that for training.

You can keep both:
- `./processed_data/` - original, for reference
- `./processed_data_v2/` - clean, for training

### Q: Can I adjust the filtering thresholds?

**A: Yes!** Edit example_usage_v2.py:

```python
# More strict filtering
train_loader.behaviors_df = filter_empty_histories(
    train_loader.behaviors_df, 
    min_history_length=5  # Require 5+ items instead of 1
)
```

Common thresholds in research:
- `min_history_length=1`: Minimal filtering (default)
- `min_history_length=5`: Standard practice
- `min_history_length=10`: Focus on experienced users

### Q: Should I use train_baseline_recommender.py or train_baseline_recommender_v2.py?

**A: Use train_baseline_recommender.py** (the original, now updated)

After running example_usage_v2.py, the original training script automatically uses clean data. The v2 training script is only needed if you want fixes applied at training time instead of preprocessing time.

---

## ✅ Summary

### Best Practice Workflow

```bash
# 1. Download MIND dataset
unzip MINDlarge_train.zip -d ./MINDlarge_train
unzip MINDlarge_dev.zip -d ./MINDlarge_dev

# 2. Process with integrated fixes
python example_usage_v2.py

# 3. Train baseline (automatically uses clean data)
python train_baseline_recommender.py

# 4. Implement diversity re-ranking (Phase 2)
# ... your diversity algorithms ...
```

### Files to Use

| File | Use It? | Purpose |
|------|---------|---------|
| example_usage_v2.py | ✅ YES | Preprocessing with fixes |
| train_baseline_recommender.py | ✅ YES | Training (updated) |
| train_baseline_recommender_v2.py | ⚠️ Optional | If you want fixes at training time |
| example_usage.py | ℹ️ Reference | Original, no fixes |

### Directory Structure

```
your-project/
├── MINDlarge_train/              # Raw data
├── MINDlarge_dev/                # Raw data
├── processed_data/               # Original (reference)
├── processed_data_v2/            # Clean (use this!)
│   ├── *.csv, *.json
│   └── popularity_scores.json    # New in v2
├── outputs/
│   └── baseline/
│       ├── baseline_recommender.pkl
│       └── baseline_results_*.json
└── *.py                          # Your code files
```

---

**Bottom line**: Run `example_usage_v2.py` once to get clean data, then use your regular training workflow. The fixes are baked into the preprocessed data! 🎯
