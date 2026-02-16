"""
Content-Based Baseline Recommender System for MIND Dataset

This module implements an accuracy-optimized content-based recommender
that serves as the baseline before applying diversity-aware re-ranking.

Key Features:
- News encoding using TF-IDF and entity embeddings
- User profile modeling from click history
- Cosine similarity-based recommendation
- Comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsEncoder:
    """
    Encodes news articles into vector representations using:
    - TF-IDF on text content
    - Entity embeddings
    - Category embeddings
    """
    
    def __init__(self, use_entities: bool = True, use_categories: bool = True):
        """
        Initialize the news encoder.
        
        Args:
            use_entities: Whether to include entity embeddings
            use_categories: Whether to include category features
        """
        self.use_entities = use_entities
        self.use_categories = use_categories
        
        # TF-IDF vectorizer for text
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.news_embeddings = None
        self.news_id_to_idx = None
        self.embedding_dim = None
        
    def fit_transform(self, news_df: pd.DataFrame, 
                     entity_embeddings: Dict = None) -> np.ndarray:
        """
        Fit the encoder and transform news articles.
        
        Args:
            news_df: DataFrame with news features
            entity_embeddings: Dictionary of entity embeddings
        
        Returns:
            News embeddings matrix (num_news, embedding_dim)
        """
        logger.info("Encoding news articles...")
        
        # Create news ID mapping
        self.news_id_to_idx = {
            news_id: idx for idx, news_id in enumerate(news_df['news_id'])
        }
        
        embeddings_list = []
        
        # 1. TF-IDF on combined text
        logger.info("Computing TF-IDF features...")
        tfidf_features = self.tfidf_vectorizer.fit_transform(
            news_df['combined_text'].fillna('')
        ).toarray()
        embeddings_list.append(tfidf_features)
        logger.info(f"TF-IDF features: {tfidf_features.shape}")
        
        # 2. Entity embeddings (averaged)
        if self.use_entities and entity_embeddings:
            logger.info("Processing entity embeddings...")
            entity_features = self._get_entity_features(
                news_df, entity_embeddings
            )
            embeddings_list.append(entity_features)
            logger.info(f"Entity features: {entity_features.shape}")
        
        # 3. Category one-hot encoding
        if self.use_categories:
            logger.info("Encoding categories...")
            category_features = self._get_category_features(news_df)
            embeddings_list.append(category_features)
            logger.info(f"Category features: {category_features.shape}")
        
        # Concatenate all features
        self.news_embeddings = np.hstack(embeddings_list)
        
        # Normalize embeddings
        self.news_embeddings = normalize(self.news_embeddings, norm='l2')
        
        self.embedding_dim = self.news_embeddings.shape[1]
        
        logger.info(f"Final news embeddings shape: {self.news_embeddings.shape}")
        
        return self.news_embeddings
    
    def _get_entity_features(self, news_df: pd.DataFrame, 
                            entity_embeddings: Dict) -> np.ndarray:
        """Extract and average entity embeddings for each news article."""
        entity_dim = 100  # MIND entity embeddings are 100-dim
        entity_features = np.zeros((len(news_df), entity_dim))
        
        for idx, row in news_df.iterrows():
            entities = row.get('all_entities', [])
            if entities and isinstance(entities, list):
                valid_embeddings = []
                for entity_id in entities:
                    if entity_id in entity_embeddings:
                        valid_embeddings.append(entity_embeddings[entity_id])
                
                if valid_embeddings:
                    entity_features[idx] = np.mean(valid_embeddings, axis=0)
        
        return entity_features
    
    def _get_category_features(self, news_df: pd.DataFrame) -> np.ndarray:
        """One-hot encode categories and subcategories."""
        # Get unique categories and subcategories
        categories = news_df['category'].unique()
        subcategories = news_df['subcategory'].unique()
        
        num_cats = len(categories)
        num_subcats = len(subcategories)
        
        cat_features = np.zeros((len(news_df), num_cats + num_subcats))
        
        cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        subcat_to_idx = {
            subcat: idx + num_cats 
            for idx, subcat in enumerate(subcategories)
        }
        
        for idx, row in news_df.iterrows():
            cat_idx = cat_to_idx[row['category']]
            subcat_idx = subcat_to_idx[row['subcategory']]
            
            cat_features[idx, cat_idx] = 1
            cat_features[idx, subcat_idx] = 1
        
        return cat_features
    
    def get_embedding(self, news_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific news article."""
        if news_id not in self.news_id_to_idx:
            return None
        idx = self.news_id_to_idx[news_id]
        return self.news_embeddings[idx]
    
    def get_embeddings_batch(self, news_ids: List[str]) -> np.ndarray:
        """Get embeddings for a batch of news articles."""
        embeddings = []
        for news_id in news_ids:
            emb = self.get_embedding(news_id)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return np.zeros((1, self.embedding_dim))
        
        return np.array(embeddings)


class UserProfiler:
    """
    Models user interests from their click history.
    """
    
    def __init__(self, news_encoder: NewsEncoder):
        """
        Initialize user profiler.
        
        Args:
            news_encoder: Fitted NewsEncoder instance
        """
        self.news_encoder = news_encoder
        self.user_profiles = {}
    
    def build_user_profile(self, history: List[str], 
                          method: str = 'average') -> np.ndarray:
        """
        Build user profile from click history.
        
        Args:
            history: List of clicked news IDs
            method: 'average', 'weighted', or 'last_n'
        
        Returns:
            User profile vector
        """
        if not history:
            return np.zeros(self.news_encoder.embedding_dim)
        
        # Get embeddings for history items
        history_embeddings = self.news_encoder.get_embeddings_batch(history)
        
        if len(history_embeddings) == 0:
            return np.zeros(self.news_encoder.embedding_dim)
        
        if method == 'average':
            # Simple average
            profile = np.mean(history_embeddings, axis=0)
        
        elif method == 'weighted':
            # Weight recent items more heavily
            weights = np.exp(np.linspace(-1, 0, len(history_embeddings)))
            weights = weights / weights.sum()
            profile = np.average(history_embeddings, axis=0, weights=weights)
        
        elif method == 'last_n':
            # Use only last N items
            n = min(10, len(history_embeddings))
            profile = np.mean(history_embeddings[-n:], axis=0)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize
        profile = normalize(profile.reshape(1, -1), norm='l2')[0]
        
        return profile
    
    def build_profiles_batch(self, user_histories: Dict[str, List[str]], 
                           method: str = 'average') -> Dict[str, np.ndarray]:
        """
        Build profiles for multiple users.
        
        Args:
            user_histories: Dictionary mapping user_id to history
            method: Profile building method
        
        Returns:
            Dictionary mapping user_id to profile vector
        """
        logger.info(f"Building profiles for {len(user_histories)} users...")
        
        profiles = {}
        for user_id, history in user_histories.items():
            profiles[user_id] = self.build_user_profile(history, method)
        
        self.user_profiles = profiles
        
        logger.info("User profiles built successfully")
        
        return profiles


class ContentBasedRecommender:
    """
    Content-based recommender using cosine similarity between
    user profiles and candidate news articles.
    """
    
    def __init__(self, news_encoder: NewsEncoder, user_profiler: UserProfiler):
        """
        Initialize recommender.
        
        Args:
            news_encoder: Fitted NewsEncoder
            user_profiler: UserProfiler instance
        """
        self.news_encoder = news_encoder
        self.user_profiler = user_profiler
    
    def score_candidates(self, user_profile: np.ndarray, 
                        candidate_ids: List[str]) -> np.ndarray:
        """
        Score candidate news articles for a user.
        
        Args:
            user_profile: User profile vector
            candidate_ids: List of candidate news IDs
        
        Returns:
            Array of relevance scores
        """
        # Get candidate embeddings
        candidate_embeddings = self.news_encoder.get_embeddings_batch(
            candidate_ids
        )
        
        # Compute cosine similarity
        scores = cosine_similarity(
            user_profile.reshape(1, -1), 
            candidate_embeddings
        )[0]
        
        return scores
    
    def recommend(self, user_id: str = None, user_profile: np.ndarray = None,
                 candidate_ids: List[str] = None, 
                 top_k: int = 10,
                 exclude_history: bool = True,
                 user_history: List[str] = None) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User identifier (if profile already built)
            user_profile: User profile vector (if not using cached)
            candidate_ids: List of candidate news IDs
            top_k: Number of recommendations
            exclude_history: Whether to exclude history items
            user_history: User's click history (for exclusion)
        
        Returns:
            List of (news_id, score) tuples
        """
        # Get user profile
        if user_profile is None:
            if user_id in self.user_profiler.user_profiles:
                user_profile = self.user_profiler.user_profiles[user_id]
            else:
                raise ValueError(
                    "Must provide either user_id or user_profile"
                )
        
        # Get all news IDs if not specified
        if candidate_ids is None:
            candidate_ids = list(self.news_encoder.news_id_to_idx.keys())
        
        # Exclude history if requested
        if exclude_history and user_history:
            history_set = set(user_history)
            candidate_ids = [
                nid for nid in candidate_ids 
                if nid not in history_set
            ]
        
        # Score candidates
        scores = self.score_candidates(user_profile, candidate_ids)
        
        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        recommendations = [
            (candidate_ids[idx], scores[idx]) 
            for idx in top_indices
        ]
        
        return recommendations
    
    def recommend_batch(self, user_data: List[Dict], 
                       top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_data: List of dicts with 'user_id', 'history', 'candidates'
            top_k: Number of recommendations per user
        
        Returns:
            List of recommendation lists
        """
        recommendations = []
        
        for data in user_data:
            user_profile = self.user_profiler.build_user_profile(
                data['history']
            )
            
            recs = self.recommend(
                user_profile=user_profile,
                candidate_ids=data.get('candidates'),
                top_k=top_k,
                exclude_history=True,
                user_history=data['history']
            )
            
            recommendations.append(recs)
        
        return recommendations


class RecommenderEvaluator:
    """
    Comprehensive evaluation metrics for the recommender system.
    """
    
    def __init__(self, recommender: ContentBasedRecommender, news_df: pd.DataFrame):
        """
        Initialize evaluator.
        
        Args:
            recommender: Trained recommender system
            news_df: DataFrame with news metadata
        """
        self.recommender = recommender
        self.news_df = news_df
        self.news_categories = dict(zip(news_df['news_id'], news_df['category']))
    
    def evaluate(self, test_data: List[Dict], 
                k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Comprehensive evaluation on test data.
        
        Args:
            test_data: List of test samples with:
                - user_id/user_idx
                - history
                - impressions (list of (news_id, label) tuples)
            k_values: List of k values for metrics
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating on {len(test_data)} test samples...")
        
        all_metrics = defaultdict(list)
        
        for sample in test_data:
            # Get user profile
            user_profile = self.recommender.user_profiler.build_user_profile(
                sample['history']
            )
            
            # Get candidates and labels from impressions
            impressions = sample['impressions']
            candidate_ids = [news_id for news_id, _ in impressions]
            labels = [label for _, label in impressions]
            
            # Get scores
            scores = self.recommender.score_candidates(
                user_profile, candidate_ids
            )
            
            # Calculate metrics for each k
            for k in k_values:
                metrics = self._calculate_metrics(
                    candidate_ids, labels, scores, k
                )
                
                for metric_name, value in metrics.items():
                    all_metrics[f"{metric_name}@{k}"].append(value)
        
        # Average metrics
        results = {
            metric: np.mean(values) 
            for metric, values in all_metrics.items()
        }
        
        # Add overall metrics (not k-dependent)
        results['auc'] = self._calculate_auc(test_data)
        results['mrr'] = np.mean(all_metrics['mrr@5'])  # Use k=5 for MRR
        
        logger.info("Evaluation complete!")
        
        return results
    
    def _calculate_metrics(self, candidate_ids: List[str], 
                          labels: List[int], scores: np.ndarray, 
                          k: int) -> Dict:
        """Calculate metrics for a single impression."""
        # Get top-k predictions
        top_k_indices = np.argsort(scores)[-k:][::-1]
        top_k_labels = [labels[idx] for idx in top_k_indices]
        
        # Precision@K
        precision = sum(top_k_labels) / k if k > 0 else 0
        
        # Recall@K
        total_relevant = sum(labels)
        recall = (
            sum(top_k_labels) / total_relevant 
            if total_relevant > 0 else 0
        )
        
        # F1@K
        f1 = (
            2 * precision * recall / (precision + recall) 
            if (precision + recall) > 0 else 0
        )
        
        # NDCG@K
        ndcg = self._calculate_ndcg(labels, scores, k)
        
        # MRR (Mean Reciprocal Rank)
        sorted_indices = np.argsort(scores)[::-1]
        mrr = 0
        for rank, idx in enumerate(sorted_indices, 1):
            if labels[idx] == 1:
                mrr = 1.0 / rank
                break
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ndcg': ndcg,
            'mrr': mrr
        }
    
    def _calculate_ndcg(self, labels: List[int], scores: np.ndarray, 
                       k: int) -> float:
        """Calculate NDCG@K."""
        # DCG
        sorted_indices = np.argsort(scores)[-k:][::-1]
        dcg = sum(
            labels[idx] / np.log2(rank + 2) 
            for rank, idx in enumerate(sorted_indices)
        )
        
        # IDCG (ideal DCG)
        ideal_labels = sorted(labels, reverse=True)[:k]
        idcg = sum(
            label / np.log2(rank + 2) 
            for rank, label in enumerate(ideal_labels)
        )
        
        return dcg / idcg if idcg > 0 else 0
    
    def _calculate_auc(self, test_data: List[Dict]) -> float:
        """Calculate AUC-ROC."""
        from sklearn.metrics import roc_auc_score
        
        all_labels = []
        all_scores = []
        
        for sample in test_data:
            user_profile = self.recommender.user_profiler.build_user_profile(
                sample['history']
            )
            
            impressions = sample['impressions']
            candidate_ids = [news_id for news_id, _ in impressions]
            labels = [label for _, label in impressions]
            
            scores = self.recommender.score_candidates(
                user_profile, candidate_ids
            )
            
            all_labels.extend(labels)
            all_scores.extend(scores)
        
        try:
            auc = roc_auc_score(all_labels, all_scores)
        except ValueError:
            auc = 0.0
        
        return auc
    
    def calculate_diversity_metrics(self, recommendations: List[List[str]], 
                                   test_data: List[Dict]) -> Dict:
        """
        Calculate diversity metrics for the recommendations.
        
        Args:
            recommendations: List of recommended news IDs for each user
            test_data: Original test data with user histories
        
        Returns:
            Dictionary with diversity metrics
        """
        logger.info("Calculating diversity metrics...")
        
        # Category diversity
        category_diversities = []
        coverage_scores = []
        gini_coefficients = []
        
        for i, recs in enumerate(recommendations):
            # Extract news IDs from (news_id, score) tuples
            rec_ids = [news_id for news_id, _ in recs]
            
            # Category diversity (unique categories / total items)
            categories = [
                self.news_categories.get(nid) 
                for nid in rec_ids 
                if nid in self.news_categories
            ]
            
            if categories:
                unique_cats = len(set(categories))
                category_diversity = unique_cats / len(categories)
                category_diversities.append(category_diversity)
                
                # Coverage
                coverage_scores.append(unique_cats)
        
        # Calculate Gini coefficient
        all_categories = []
        for recs in recommendations:
            rec_ids = [news_id for news_id, _ in recs]
            all_categories.extend([
                self.news_categories.get(nid) 
                for nid in rec_ids 
                if nid in self.news_categories
            ])
        
        gini = self._calculate_gini(all_categories)
        
        return {
            'avg_category_diversity': np.mean(category_diversities),
            'avg_coverage': np.mean(coverage_scores),
            'gini_coefficient': gini
        }
    
    def _calculate_gini(self, categories: List[str]) -> float:
        """Calculate Gini coefficient for category distribution."""
        if not categories:
            return 0
        
        # Count frequencies
        from collections import Counter
        counts = Counter(categories)
        values = np.array(list(counts.values()))
        
        # Sort values
        values = np.sort(values)
        n = len(values)
        
        # Calculate Gini
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
        
        return gini


def save_model(recommender: ContentBasedRecommender, filepath: str):
    """Save trained recommender model."""
    model_data = {
        'news_encoder': recommender.news_encoder,
        'user_profiler': recommender.user_profiler
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> ContentBasedRecommender:
    """Load trained recommender model."""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    recommender = ContentBasedRecommender(
        model_data['news_encoder'],
        model_data['user_profiler']
    )
    
    logger.info(f"Model loaded from {filepath}")
    
    return recommender


if __name__ == "__main__":
    # Example usage
    print("Content-Based Recommender - Example")
    print("=" * 60)
    
    # This would be called from the main training script
    # See train_baseline_recommender.py for full example
