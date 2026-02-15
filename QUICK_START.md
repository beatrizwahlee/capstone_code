# QUICK START GUIDE
## MIND Dataset Preprocessing for Diversity-Aware News Recommender

**Project**: Capstone - Diversity-Aware Personalized News Recommender System  
**Student**: [Your Name]  
**Date**: February 2026

---

## ⚡ Get Started in 5 Minutes

### Step 1: Install Dependencies (1 min)
```bash
pip install pandas numpy
```

### Step 2: Download MIND Dataset (2 min)
Visit: https://msnews.github.io/

Download:
- MINDlarge_train.zip (or MINDsmall_train.zip for testing)
- MINDlarge_dev.zip (or MINDsmall_dev.zip)

### Step 3: Extract Data (1 min)
```bash
unzip MINDlarge_train.zip -d ./MINDlarge_train
unzip MINDlarge_dev.zip -d ./MINDlarge_dev
```

### Step 4: Run Preprocessing (1 min)
```bash
python example_usage.py
```

Done! Your data is now loaded and preprocessed in `./processed_data/`

---

## 📂 What You Have Now

### Core Modules (Python Files)

1. **mind_data_loader.py** - Main data loading and preprocessing
   - `MINDDataLoader` class: Loads raw MIND files
   - `MINDPreprocessor` class: Cleans and prepares data
   
2. **data_quality_checker.py** - Data validation
   - `DataQualityChecker` class: Validates data quality
   - Generates diagnostic reports
   
3. **example_usage.py** - Complete working example
   - End-to-end pipeline
   - Shows how to use all components
   
4. **config.py** - Configuration management
   - Centralized settings
   - Multiple preset configs
   - Easy parameter tuning

5. **requirements.txt** - Python dependencies
   - Minimal requirements for data processing
   - Optional packages for future steps

6. **README.md** - Comprehensive documentation
   - Detailed usage instructions
   - Troubleshooting guide
   - Next steps for your project

---

## 🎯 Critical Information for Your Project

### Data Format You Need

For your diversity-aware recommender, you'll need:

#### 1. **News Features** (for encoding with NLP)
```python
# From news_features_train.csv
- news_id: Unique identifier
- title: Article title (encode with BERT/SBERT)
- abstract: Article abstract (encode with BERT/SBERT)
- category: News category (for diversity)
- subcategory: Subcategory (for fine-grained diversity)
- all_entities: WikiData entities (for knowledge graph)
- entity_embedding: Pre-computed entity vectors (100-dim)
```

#### 2. **User Interactions** (for training)
```python
# From train_data list
- user_idx: User identifier
- news_idx: News identifier
- label: Click (1) or not (0)
- history_indices: User's click history
- time: Timestamp (for temporal analysis)
```

#### 3. **Diversity Statistics** (for re-ranking)
```python
# From diversity_stats.json
- category_distribution: Overall category breakdown
- user_category_diversity: Each user's category preferences
- subcategory_distribution: Subcategory breakdown
```

---

## 🔄 Your Recommendation Pipeline

### Phase 1: ✅ DONE - Data Loading
You have:
- Clean, parsed data
- Quality checks passed
- Proper train/validation splits
- Diversity statistics computed

### Phase 2: NEXT - Text Encoding

**What to do**: Encode news titles/abstracts into embeddings

**Recommended approach**:
```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode news
news_embeddings = model.encode(
    news_df['combined_text'].tolist(),
    show_progress_bar=True
)

# Shape: (num_news, 384) - 384-dim embeddings
```

**Alternative models**:
- `all-mpnet-base-v2` - Higher quality, slower
- `distilbert-base-nli-stsb-mean-tokens` - Lighter
- Fine-tune BERT on news domain data

### Phase 3: User Modeling

**Goal**: Create user profile from their click history

**Approach 1 - Simple Average**:
```python
user_profile = np.mean([news_embeddings[idx] for idx in history], axis=0)
```

**Approach 2 - Attention-Weighted** (Better):
```python
# Weight recent clicks more
weights = softmax([1/sqrt(pos) for pos in range(1, len(history)+1)])
user_profile = np.average(
    [news_embeddings[idx] for idx in history],
    weights=weights,
    axis=0
)
```

### Phase 4: Relevance Scoring

**Goal**: Predict click probability

**Simple approach** (Good baseline):
```python
# Cosine similarity
relevance_score = cosine_similarity(
    user_profile.reshape(1, -1),
    candidate_news_embedding.reshape(1, -1)
)[0][0]
```

**Neural approach** (Better):
```python
# Feed-forward network
class RelevanceModel(nn.Module):
    def __init__(self, embedding_dim=384):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, user_emb, news_emb):
        x = torch.cat([user_emb, news_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))
```

### Phase 5: Diversity Re-Ranking

**Goal**: Re-rank top-K candidates to maximize diversity while maintaining relevance

**MMR Algorithm** (Maximal Marginal Relevance):
```python
def mmr_rerank(candidates, scores, lambda_param=0.5):
    """
    candidates: List of candidate news items
    scores: Relevance scores
    lambda_param: 0=pure diversity, 1=pure relevance
    """
    selected = []
    remaining = list(range(len(candidates)))
    
    # Select first item (highest relevance)
    first = np.argmax(scores)
    selected.append(candidates[first])
    remaining.remove(first)
    
    while remaining and len(selected) < top_k:
        mmr_scores = []
        for idx in remaining:
            relevance = scores[idx]
            
            # Max similarity to already selected items
            max_sim = max([
                category_similarity(candidates[idx], s) 
                for s in selected
            ])
            
            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append(mmr)
        
        # Select item with highest MMR
        best = remaining[np.argmax(mmr_scores)]
        selected.append(candidates[best])
        remaining.remove(best)
    
    return selected
```

**Calibration** (Match user's category preferences):
```python
def calibrated_rerank(candidates, user_category_dist):
    """
    Ensure recommended categories match user's historical distribution
    """
    recommended_dist = {}
    selected = []
    
    for candidate in sorted(candidates, key=lambda x: x.score, reverse=True):
        category = candidate.category
        
        # Check if adding this would improve calibration
        target_count = user_category_dist[category] * len(selected_so_far)
        current_count = recommended_dist.get(category, 0)
        
        if current_count < target_count:
            selected.append(candidate)
            recommended_dist[category] = current_count + 1
    
    return selected
```

---

## 📊 Evaluation Metrics to Implement

### Accuracy Metrics
```python
# 1. AUC (Area Under ROC Curve)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(true_labels, predicted_scores)

# 2. MRR (Mean Reciprocal Rank)
def mrr(ranked_list, clicked_items):
    for rank, item in enumerate(ranked_list, 1):
        if item in clicked_items:
            return 1.0 / rank
    return 0.0

# 3. nDCG@K (Normalized Discounted Cumulative Gain)
from sklearn.metrics import ndcg_score
ndcg = ndcg_score([true_relevance], [predicted_relevance], k=10)
```

### Diversity Metrics
```python
# 1. ILD (Intra-List Diversity)
def ild(ranked_list):
    """Average dissimilarity between all pairs"""
    n = len(ranked_list)
    total_dissim = 0
    for i in range(n):
        for j in range(i+1, n):
            total_dissim += dissimilarity(ranked_list[i], ranked_list[j])
    return total_dissim / (n * (n-1) / 2)

# 2. Category Coverage
def category_coverage(ranked_list, all_categories):
    """Percentage of categories represented"""
    represented = set(item.category for item in ranked_list)
    return len(represented) / len(all_categories)

# 3. Entropy
def category_entropy(ranked_list):
    """Distribution uniformity"""
    category_counts = Counter(item.category for item in ranked_list)
    probs = [count / len(ranked_list) for count in category_counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

# 4. Calibration
def calibration_error(recommended_dist, user_historical_dist):
    """KL-divergence between distributions"""
    kl_div = sum(
        user_historical_dist[cat] * np.log(
            user_historical_dist[cat] / recommended_dist.get(cat, 1e-10)
        )
        for cat in user_historical_dist
    )
    return kl_div
```

---

## 🎓 Key Insights for Your Write-Up

### 1. Problem Statement
- **Echo chambers** form when users only see content aligned with existing preferences
- **Challenge**: Balance relevance (what users click) with diversity (exposure to variety)
- **Your approach**: Multi-objective re-ranking that explicitly optimizes for both

### 2. Dataset Characteristics
Use `diversity_stats.json` to report:
- Number of categories and their distribution
- User diversity patterns (from `user_category_diversity`)
- Entity coverage and richness
- Temporal patterns in the data

### 3. Diversity Dimensions
Your system addresses multiple diversity facets:
- **Category diversity**: Variety of news topics
- **Provider diversity**: Multiple news sources (extract from URL)
- **Viewpoint diversity**: Different perspectives (use entity co-occurrence)
- **Calibration**: Matching user preferences while expanding exposure

### 4. Trade-off Analysis
Document the accuracy-diversity trade-off:
- Plot: Diversity weight (0 to 1) vs. AUC and ILD
- Show: Pareto frontier of accuracy and diversity metrics
- Discuss: Optimal operating points for different user segments

---

## 🔍 Common Pitfalls to Avoid

### 1. Data Leakage
❌ **Don't**: Use future news in user history  
✅ **Do**: Ensure history only contains news before impression time

### 2. Cold Start
❌ **Don't**: Ignore users with short history  
✅ **Do**: Have fallback strategy (popular items, category-based)

### 3. Overfitting to Diversity
❌ **Don't**: Maximize diversity at expense of all relevance  
✅ **Do**: Use tunable diversity weight, validate on user satisfaction

### 4. Category Imbalance
❌ **Don't**: Only show majority categories even with diversity  
✅ **Do**: Ensure minority categories get fair representation

---

## 📝 Your Next Actions (Priority Order)

1. **Today**: Run `python example_usage.py` - verify everything works
2. **This week**: Implement text encoding with Sentence-BERT
3. **Next week**: Build simple baseline (cosine similarity)
4. **Week 3**: Implement neural relevance model
5. **Week 4**: Add MMR re-ranking
6. **Week 5**: Add calibration and other diversity mechanisms
7. **Week 6**: Comprehensive evaluation and analysis
8. **Week 7**: Write-up and polish

---

## 💡 Pro Tips

1. **Start simple**: Get cosine similarity baseline working first
2. **Iterate quickly**: Use MINDsmall for fast experiments
3. **Track everything**: Log all experiments with parameters and results
4. **Visualize**: Plot diversity vs. relevance for different parameters
5. **User studies**: If possible, get human evaluation of diversity

---

## 📚 Resources

- **MIND Paper**: https://arxiv.org/abs/2010.00527
- **Sentence-BERT**: https://www.sbert.net/
- **MMR Algorithm**: Carbonell & Goldstein (1998)
- **Calibration**: Steck (2018) - "Calibrated Recommendations"

---

## ✅ Checklist

Before moving to next phase:
- [ ] Data loaded successfully
- [ ] Quality checks passed
- [ ] Processed data saved
- [ ] Diversity statistics computed
- [ ] Configuration file reviewed
- [ ] Understand data format
- [ ] Next steps planned

---

Good luck with your capstone! You have a solid foundation to build on. 🚀

**Remember**: The goal is not perfect accuracy, but finding the optimal balance between relevance and diversity that creates a better user experience and combats echo chambers.
