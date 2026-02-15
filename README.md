# MIND Dataset Loader & Preprocessor
## Diversity-Aware News Recommender System - Capstone Project

This repository contains the data loading and preprocessing pipeline for your diversity-aware personalized news recommender system capstone project.

## 📋 Overview

The code processes the Microsoft News Dataset (MIND) and prepares it for building an end-to-end news recommendation pipeline that:
1. Encodes news with NLP embeddings
2. Generates relevance-optimized recommendations
3. Applies multi-objective re-ranking to balance accuracy with diversity, calibration, and fairness

## 🗂️ Dataset Structure

The MIND dataset contains:
- **behaviors.tsv**: User click histories and impression logs
- **news.tsv**: News article metadata (category, title, abstract, entities)
- **entity_embedding.vec**: 100-dimensional entity embeddings from WikiData
- **relation_embedding.vec**: Relation embeddings from knowledge graph

## 🚀 Quick Start

### 1. Installation

```bash
# Clone/download this repository
cd your-project-directory

# Install dependencies
pip install -r requirements.txt
```

### 2. Download MIND Dataset

Download the MIND dataset from: https://msnews.github.io/

You can use either:
- **MINDsmall**: Smaller dataset for quick testing (~50MB)
- **MINDlarge**: Full dataset for final results (~500MB)

Download both training and validation sets.

### 3. Organize Your Data

Place the extracted data in the following structure:
```
your-project/
├── MINDlarge_train/
│   ├── behaviors.tsv
│   ├── news.tsv
│   ├── entity_embedding.vec
│   └── relation_embedding.vec
├── MINDlarge_dev/
│   ├── behaviors.tsv
│   ├── news.tsv
│   ├── entity_embedding.vec
│   └── relation_embedding.vec
├── mind_data_loader.py
├── data_quality_checker.py
└── example_usage.py
```

### 4. Run the Preprocessing Pipeline

```bash
python example_usage.py
```

This will:
- Load training and validation data
- Run data quality checks
- Preprocess and clean the data
- Generate statistics and summaries
- Save processed data to `./processed_data/`

## 📁 Module Descriptions

### `mind_data_loader.py`
Main module containing:
- **MINDDataLoader**: Loads raw MIND dataset files
  - Parses behaviors.tsv (user interactions)
  - Parses news.tsv (article metadata)
  - Loads entity and relation embeddings
  
- **MINDPreprocessor**: Preprocesses data for modeling
  - Encodes categories and subcategories
  - Creates ID mappings
  - Builds user-item interaction matrices
  - Extracts news features
  - Calculates diversity statistics

### `data_quality_checker.py`
Quality assurance module:
- Checks for missing data
- Detects duplicates
- Validates entity embedding coverage
- Analyzes category balance
- Checks temporal consistency
- Evaluates interaction quality
- Generates comprehensive quality reports

### `example_usage.py`
End-to-end example script demonstrating:
- Complete data loading workflow
- Quality checking procedures
- Data preprocessing steps
- Data export and saving
- Next steps for your project

## 🎯 What You Get After Preprocessing

The processed data includes:

### 1. News Features (`news_features_train.csv`)
- News ID, category, subcategory
- Encoded categories
- Title and abstract text
- Entity information
- Text length statistics
- Averaged entity embeddings

### 2. Diversity Statistics (`diversity_stats.json`)
- Category distribution across dataset
- Subcategory distribution
- Entity diversity per category
- User category diversity metrics
- Used for calibration and fairness evaluation

### 3. Encoders (`encoders.json`)
- Category name → index mapping
- Subcategory name → index mapping
- Needed for consistent encoding across train/test

### 4. Interaction Data (`sample_train_interactions.csv`)
- User ID and index
- News ID and index
- Click label (0 or 1)
- User history
- Timestamp information

### 5. Quality Report (`data_quality_report.txt`)
- Data quality assessment
- Identified issues
- Coverage statistics

## 🔧 Customization

### Adjusting Data Paths

Edit `example_usage.py`:
```python
TRAIN_DATA_DIR = "./path/to/your/train/data"
VALID_DATA_DIR = "./path/to/your/valid/data"
```

### Processing Specific Components

```python
from mind_data_loader import MINDDataLoader, MINDPreprocessor

# Load only specific components
loader = MINDDataLoader("./MINDlarge_train", dataset_type='train')
loader.behaviors_df = loader.load_behaviors()
loader.news_df = loader.load_news()

# Run specific preprocessing steps
preprocessor = MINDPreprocessor(loader)
preprocessor.encode_categories()
news_features = preprocessor.extract_news_features()
```

## 📊 Data Format Details

### Behaviors DataFrame
| Column | Type | Description |
|--------|------|-------------|
| impression_id | int | Unique impression ID |
| user_id | str | Anonymous user ID |
| time | datetime | Impression timestamp |
| history | list[str] | List of clicked news IDs (ordered) |
| impressions | list[tuple] | List of (news_id, label) pairs |

### News Features DataFrame
| Column | Type | Description |
|--------|------|-------------|
| news_id | str | Unique news ID |
| category | str | News category |
| subcategory | str | News subcategory |
| category_encoded | int | Encoded category index |
| subcategory_encoded | int | Encoded subcategory index |
| title | str | News title |
| abstract | str | News abstract |
| combined_text | str | Title + abstract |
| all_entities | list[str] | WikiData entity IDs |
| entity_count | int | Number of entities |
| entity_embedding | ndarray | Averaged entity embedding |

### Interaction Data Format
```python
{
    'user_id': 'U12345',
    'user_idx': 0,
    'news_id': 'N54321',
    'news_idx': 42,
    'label': 1,  # 1 = clicked, 0 = not clicked
    'history': ['N1', 'N2', 'N3'],
    'history_indices': [0, 1, 2],
    'impression_id': 91,
    'time': Timestamp('2019-11-15 10:22:32')
}
```

## 🎓 Next Steps for Your Capstone

### Phase 1: NLP Embeddings (Current Phase)
✅ Data loading complete  
✅ Data preprocessing complete  
→ **Next**: Implement text encoding

**Recommendations**:
- Use **Sentence-BERT** for semantic embeddings
- Consider **BERT** or **RoBERTa** fine-tuned on news
- Try **News-specific models** like NewsEmbed

```python
# Pseudocode for next step
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
news_embeddings = model.encode(news_df['combined_text'].tolist())
```

### Phase 2: User Modeling
- Aggregate news embeddings from user history
- Implement attention mechanisms for weighting recent vs. old clicks
- Build user interest profiles

### Phase 3: Relevance Model
- Build neural click prediction model
- Architecture: User embedding + News embedding → Click probability
- Train on click/no-click labels from impressions

### Phase 4: Diversity-Aware Re-ranking
**Key diversity metrics to implement**:

1. **Category Diversity**: Ensure variety of news categories
   - Use `diversity_stats['category_distribution']`
   - MMR (Maximal Marginal Relevance)
   - ILS (Intra-List Similarity)

2. **Calibration**: Match user's historical preferences
   - Use `diversity_stats['user_category_diversity']`
   - Compare recommended category dist. to user's history

3. **Provider Fairness**: Balance across news sources
   - Extract provider from URL domain
   - Ensure minority providers get representation

4. **Viewpoint Diversity**: Balance political perspectives
   - May need external viewpoint labels
   - Consider using entity co-occurrence patterns

### Phase 5: Evaluation
**Accuracy Metrics**:
- AUC (Area Under ROC Curve)
- MRR (Mean Reciprocal Rank)
- nDCG (Normalized Discounted Cumulative Gain)

**Diversity Metrics**:
- ILD (Intra-List Diversity)
- Coverage (% of categories represented)
- Entropy (distribution uniformity)
- Gini coefficient (fairness)

**Combined Metrics**:
- Weighted combination of accuracy and diversity
- Pareto frontier analysis

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: behaviors.tsv not found`
- **Solution**: Check that your data directory path is correct and contains extracted .tsv files

**Issue**: `MemoryError when loading large dataset`
- **Solution**: Start with MINDsmall, or process data in chunks:
```python
# Process behaviors in chunks
chunk_size = 10000
for chunk in pd.read_csv('behaviors.tsv', sep='\t', chunksize=chunk_size):
    # Process chunk
    pass
```

**Issue**: Low entity embedding coverage
- **Solution**: This is normal - some entities may not have embeddings in WikiData subset. Handle missing embeddings gracefully:
```python
# Use zero vector or average embedding for missing entities
if entity_id not in entity_embeddings:
    embedding = np.zeros(100)  # or use average
```

## 📚 Key Papers & Resources

1. **MIND Dataset Paper**: Wu et al. "MIND: A Large-scale Dataset for News Recommendation" (ACL 2020)
2. **Diversity in Recommendations**: Kunaver & Požrl "Diversity in recommender systems – A survey" (2017)
3. **Fairness in Ranking**: Zehlike et al. "FA*IR: A Fair Top-k Ranking Algorithm" (2017)
4. **Neural News Recommendation**: Wu et al. "Neural News Recommendation with Multi-Head Self-Attention" (EMNLP 2019)

## 🤝 Project Structure Recommendations

```
capstone-project/
├── data/
│   ├── raw/              # Original MIND data
│   └── processed/        # Preprocessed data
├── src/
│   ├── data/            # Data loading (this code)
│   ├── models/          # Recommendation models
│   ├── reranking/       # Diversity re-ranking
│   └── evaluation/      # Metrics and evaluation
├── notebooks/           # Jupyter notebooks for experiments
├── experiments/         # Experiment configs and results
└── docs/               # Project documentation
```

## 📝 Citation

If you use this code in your capstone or research, please cite the MIND dataset:

```bibtex
@inproceedings{wu2020mind,
  title={Mind: A large-scale dataset for news recommendation},
  author={Wu, Fangzhao and Qiao, Ying and Chen, Jiun-Hung and Wu, Chuhan and Qi, Tao and Lian, Jianxun and Liu, Danyang and Xie, Xing and Gao, Jianfeng and Wu, Winnie and others},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={3597--3606},
  year={2020}
}
```

## 📧 Support

For questions about this code:
- Check the troubleshooting section
- Review the example_usage.py for usage patterns
- Examine the quality report for data issues

For questions about the MIND dataset:
- Visit: https://msnews.github.io/
- GitHub: https://github.com/msnews/msnews.github.io

---

Good luck with your capstone project! 🎓
