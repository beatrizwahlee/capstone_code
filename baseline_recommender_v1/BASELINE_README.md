# Baseline Content-Based Recommender System

## Overview

This is the **accuracy-optimized baseline** for your diversity-aware news recommender system capstone project. It implements a content-based filtering approach using:

- **TF-IDF** text features from news titles and abstracts
- **Entity embeddings** from WikiData knowledge graph (100-dim)
- **Category features** (one-hot encoded categories and subcategories)
- **Cosine similarity** for relevance scoring

This baseline achieves strong accuracy metrics before applying diversity-aware re-ranking.

## 📁 Files

### Core Modules

1. **content_based_recommender.py** - Main recommender implementation
   - `NewsEncoder`: Encodes news articles into vectors
   - `UserProfiler`: Models user interests from click history
   - `ContentBasedRecommender`: Generates recommendations
   - `RecommenderEvaluator`: Comprehensive evaluation metrics

2. **train_baseline_recommender.py** - Quick training with preprocessed data
   - Uses the processed data from `example_usage.py`
   - Good for testing and development

3. **train_with_full_dataset.py** - Full training pipeline
   - Loads raw MIND dataset
   - Processes and trains on complete data
   - Use this for final results

## 🚀 Quick Start

### Option 1: Train with Pre-processed Data (Fast)

```bash
# First, ensure you've run the data preprocessing
python example_usage.py

# Then train the baseline
python train_baseline_recommender.py
```

This uses the processed data from `./processed_data/` directory.

### Option 2: Train with Full Dataset (Complete)

```bash
# Download MIND dataset first from https://msnews.github.io/

# Train on full dataset
python train_with_full_dataset.py \
    --train_dir ./MINDlarge_train \
    --valid_dir ./MINDlarge_dev \
    --output_dir ./outputs/baseline
```

## 📊 Evaluation Metrics

The recommender is evaluated using:

### Accuracy Metrics

1. **AUC (Area Under ROC Curve)**
   - Measures overall ranking quality
   - Higher is better (max = 1.0)

2. **MRR (Mean Reciprocal Rank)**
   - Average of 1/rank for first relevant item
   - Higher is better (max = 1.0)

3. **Precision@K**
   - Fraction of top-K that are relevant
   - `Precision@K = (Relevant in top-K) / K`

4. **Recall@K**
   - Fraction of relevant items in top-K
   - `Recall@K = (Relevant in top-K) / (Total relevant)`

5. **F1@K**
   - Harmonic mean of Precision and Recall
   - `F1@K = 2 * (Precision * Recall) / (Precision + Recall)`

6. **NDCG@K (Normalized Discounted Cumulative Gain)**
   - Measures ranking quality with position discount
   - Higher positions contribute more
   - Higher is better (max = 1.0)

### Diversity Metrics (Baseline)

1. **Category Diversity**
   - Average ratio of unique categories in recommendations
   - `Diversity = Unique categories / Total items`

2. **Coverage**
   - Number of unique categories represented
   - Higher means more diverse content

3. **Gini Coefficient**
   - Measures inequality in category distribution
   - 0 = perfectly equal, 1 = perfectly unequal
   - Lower is more fair

## 🎯 Expected Performance

Based on MIND dataset benchmarks, you should expect:

| Metric | Expected Range | Good Performance |
|--------|---------------|------------------|
| AUC | 0.60 - 0.70 | > 0.65 |
| MRR | 0.25 - 0.35 | > 0.30 |
| NDCG@10 | 0.30 - 0.40 | > 0.35 |
| Precision@10 | 0.20 - 0.30 | > 0.25 |

**Note**: These are for content-based baseline. More sophisticated models (e.g., neural collaborative filtering, NRMS) achieve higher scores.

## 🔧 How It Works

### 1. News Encoding

```python
news_encoder = NewsEncoder(use_entities=True, use_categories=True)
news_embeddings = news_encoder.fit_transform(news_df, entity_embeddings)
```

**Feature Components**:
- TF-IDF vectors (5000 dimensions) from title + abstract
- Entity embeddings (100 dimensions) - averaged from WikiData
- Category one-hot (18 categories + 285 subcategories)

**Final embedding**: ~5,300 dimensions, L2-normalized

### 2. User Profile Modeling

```python
user_profiler = UserProfiler(news_encoder)
user_profile = user_profiler.build_user_profile(history, method='weighted')
```

**Methods**:
- `'average'`: Simple mean of history embeddings
- `'weighted'`: Recent items weighted more (exponential decay)
- `'last_n'`: Use only last N items

**Default**: `'weighted'` for better recency modeling

### 3. Recommendation

```python
recommender = ContentBasedRecommender(news_encoder, user_profiler)
recommendations = recommender.recommend(
    user_profile=user_profile,
    candidate_ids=all_news_ids,
    top_k=10
)
```

**Scoring**: Cosine similarity between user profile and candidate news

```
score(u, n) = cos(profile_u, embedding_n)
              = (profile_u · embedding_n) / (||profile_u|| × ||embedding_n||)
```

### 4. Evaluation

```python
evaluator = RecommenderEvaluator(recommender, news_df)
results = evaluator.evaluate(test_data, k_values=[5, 10, 20])
```

Computes all metrics on held-out test data.

## 📈 Understanding Your Results

### Example Output

```
EVALUATION RESULTS
================================================================================

--- ACCURACY METRICS ---

Overall:
  AUC:                 0.6521
  MRR:                 0.3124

@5:
  Precision@5:        0.2840
  Recall@5:           0.1420
  F1@5:               0.1893
  NDCG@5:             0.3256

@10:
  Precision@10:       0.2340
  Recall@10:          0.2340
  F1@10:              0.2340
  NDCG@10:            0.3567

--- DIVERSITY METRICS ---
  Avg Category Diversity:  0.4532
  Avg Coverage:            4.53
  Gini Coefficient:        0.6234
```

### Interpretation

**Good Signs**:
- AUC > 0.65: Model ranks relevant items higher
- NDCG@10 > 0.35: Good ranking quality
- Precision@10 > 0.20: 1 in 5 recommendations are clicked

**Areas for Improvement**:
- Category Diversity < 0.5: Echo chamber effect
- Gini > 0.6: Unequal category representation
- This is why we need diversity-aware re-ranking!

## 🔍 Analyzing Your Model

### View Top Features

```python
# Most important TF-IDF features
feature_names = news_encoder.tfidf_vectorizer.get_feature_names_out()
top_features = np.argsort(np.abs(news_embeddings[:, :5000]).sum(axis=0))[-20:]
print("Top TF-IDF features:", [feature_names[i] for i in top_features])
```

### Check User Profile

```python
user_profile = user_profiler.build_user_profile(['N12345', 'N67890'])
print(f"Profile shape: {user_profile.shape}")
print(f"Profile norm: {np.linalg.norm(user_profile)}")  # Should be ~1.0
```

### Inspect Recommendations

```python
recs = recommender.recommend(
    user_profile=user_profile,
    top_k=10
)

for news_id, score in recs:
    news_info = news_df[news_df['news_id'] == news_id].iloc[0]
    print(f"{news_id}: {score:.4f} - {news_info['title']}")
    print(f"  Category: {news_info['category']}")
```

## 🎓 Next Steps for Your Capstone

### Phase 1: ✅ DONE - Baseline Recommender
- Content-based model implemented
- Comprehensive evaluation metrics
- Baseline performance established

### Phase 2: Diversity-Aware Re-ranking

Implement these algorithms on top of your baseline:

#### A. MMR (Maximal Marginal Relevance)
```python
def mmr_rerank(candidates, scores, lambda_param=0.5):
    # Balance relevance (scores) with diversity (dissimilarity)
    # See: Carbonell & Goldstein (1998)
    pass
```

#### B. xQuAD (eXplicit Query Aspect Diversification)
```python
def xquad_rerank(candidates, scores, category_dist):
    # Maximize coverage of categories proportional to user's interests
    # See: Santos et al. (2010)
    pass
```

#### C. PM-2 (Proportionality-based)
```python
def pm2_rerank(candidates, scores, category_dist):
    # Ensure proportional representation of categories
    # See: Dwork et al. (2012)
    pass
```

#### D. Calibration-based
```python
def calibrated_rerank(candidates, scores, user_category_dist):
    # Match user's historical category distribution
    # See: Steck (2018)
    pass
```

### Phase 3: Comparative Analysis

Compare:
- **Baseline** (pure relevance)
- **Each diversity algorithm** at different λ values
- **Combined approach** (multiple objectives)

Create plots:
- Accuracy vs. Diversity (Pareto frontier)
- Per-category fairness
- User satisfaction by diversity preference

### Phase 4: User Study (Optional)

If time permits:
- Show recommendations to real users
- Measure satisfaction and perceived diversity
- Validate that diversity improves experience

## 📚 Key References

1. **Content-Based Filtering**
   - Lops, P., et al. (2011). "Content-based Recommender Systems"

2. **Evaluation Metrics**
   - Herlocker, J. L., et al. (2004). "Evaluating collaborative filtering recommender systems"
   - Shani, G., & Gunawardana, A. (2011). "Evaluating recommendation systems"

3. **Diversity in Recommendations**
   - Kunaver, M., & Požrl, T. (2017). "Diversity in recommender systems – A survey"
   - Raza, S., & Ding, C. (2024). "News Recommender System: A Review"

4. **Fairness Metrics**
   - Gini Coefficient: Measurement of statistical dispersion
   - Mehrotra, R., et al. (2018). "Towards a Fair Marketplace"

## 🐛 Troubleshooting

### Issue: Low AUC (<0.55)

**Possible causes**:
- Insufficient training data
- Poor text preprocessing
- Missing entity embeddings

**Solutions**:
```python
# Use more features
news_encoder = NewsEncoder(
    use_entities=True,
    use_categories=True
)

# Try different TF-IDF params
news_encoder.tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,  # Increase
    ngram_range=(1, 3)   # Add trigrams
)
```

### Issue: Memory Error

**Solution**: Process in batches
```python
# In train_with_full_dataset.py
# Evaluate on subset
val_data_subset = val_data[:10000]
results = evaluator.evaluate(val_data_subset)
```

### Issue: Slow Training

**Solution**: Use TF-IDF only
```python
news_encoder = NewsEncoder(
    use_entities=False,  # Disable
    use_categories=True
)
```

## 💡 Pro Tips

1. **Start with MINDsmall** for fast iteration
2. **Save intermediate results** to avoid recomputation
3. **Profile your code** to find bottlenecks
4. **Visualize embeddings** with t-SNE/UMAP
5. **Analyze errors**: Which types of news are hard to recommend?

## 📧 Questions?

This baseline provides a solid foundation. Focus on:
1. Getting it working with your data
2. Understanding the metrics
3. Establishing performance benchmarks
4. Then move to diversity re-ranking

Good luck with your capstone! 🎓
