"""
MIND Dataset Loader and Preprocessor
For Diversity-Aware News Recommender System

This module handles loading, cleaning, and preprocessing the MIND dataset
for a personalized news recommendation system with diversity considerations.
"""

import pandas as pd
import numpy as np
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import re
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MINDDataLoader:
    """
    Loader for MIND (Microsoft News Dataset) with preprocessing
    for diversity-aware recommendation.
    """
    
    def __init__(self, data_dir: str, dataset_type: str = 'train'):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to directory containing MIND dataset
            dataset_type: 'train' or 'valid'
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        
        # Data containers
        self.behaviors_df = None
        self.news_df = None
        self.entity_embeddings = None
        self.relation_embeddings = None
        
        # Processed data
        self.user_histories = None
        self.news_features = None
        self.category_mapping = None
        self.subcategory_mapping = None
        
    def load_all_data(self, extract_zip: bool = False, zip_path: Optional[str] = None):
        """
        Load all components of the MIND dataset.
        
        Args:
            extract_zip: Whether to extract from zip file first
            zip_path: Path to zip file if extract_zip is True
        """
        if extract_zip and zip_path:
            self._extract_zip(zip_path)
        
        logger.info(f"Loading {self.dataset_type} dataset...")
        
        # Load each component
        self.behaviors_df = self.load_behaviors()
        self.news_df = self.load_news()
        self.entity_embeddings = self.load_entity_embeddings()
        self.relation_embeddings = self.load_relation_embeddings()
        
        logger.info("All data loaded successfully!")
        return self
    
    def _extract_zip(self, zip_path: str):
        """Extract zip file to data directory."""
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        logger.info("Extraction complete!")
    
    def load_behaviors(self) -> pd.DataFrame:
        """
        Load and parse behaviors.tsv file.
        
        Returns:
            DataFrame with columns: impression_id, user_id, time, history, impressions
        """
        behaviors_path = self.data_dir / 'behaviors.tsv'
        logger.info(f"Loading behaviors from {behaviors_path}...")
        
        df = pd.read_csv(
            behaviors_path,
            sep='\t',
            header=None,
            names=['impression_id', 'user_id', 'time', 'history', 'impressions']
        )
        
        # Parse time to datetime
        df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y %I:%M:%S %p')
        
        # Parse history (space-separated news IDs)
        df['history'] = df['history'].fillna('').apply(
            lambda x: x.split() if x else []
        )
        
        # Parse impressions (format: NewsID-Label NewsID-Label ...)
        df['impressions'] = df['impressions'].apply(self._parse_impressions)
        
        logger.info(f"Loaded {len(df)} behavior records")
        logger.info(f"Unique users: {df['user_id'].nunique()}")
        
        return df
    
    def _parse_impressions(self, impressions_str: str) -> List[Tuple[str, int]]:
        """
        Parse impressions string into list of (news_id, label) tuples.
        
        Args:
            impressions_str: String like "N129416-0 N26703-1 N120089-1"
        
        Returns:
            List of tuples: [('N129416', 0), ('N26703', 1), ...]
        """
        if pd.isna(impressions_str) or not impressions_str:
            return []
        
        parsed = []
        for item in impressions_str.split():
            news_id, label = item.rsplit('-', 1)
            parsed.append((news_id, int(label)))
        
        return parsed
    
    def load_news(self) -> pd.DataFrame:
        """
        Load and parse news.tsv file.
        
        Returns:
            DataFrame with news metadata and entities
        """
        news_path = self.data_dir / 'news.tsv'
        logger.info(f"Loading news from {news_path}...")
        
        df = pd.read_csv(
            news_path,
            sep='\t',
            header=None,
            names=[
                'news_id', 'category', 'subcategory', 'title', 
                'abstract', 'url', 'title_entities', 'abstract_entities'
            ]
        )
        
        # Handle missing values
        df['abstract'] = df['abstract'].fillna('')
        df['title_entities'] = df['title_entities'].fillna('[]')
        df['abstract_entities'] = df['abstract_entities'].fillna('[]')
        
        # Parse entity JSON strings
        df['title_entities'] = df['title_entities'].apply(self._parse_entities)
        df['abstract_entities'] = df['abstract_entities'].apply(self._parse_entities)
        
        # Create combined text field
        df['combined_text'] = df['title'] + ' ' + df['abstract']
        
        # Extract all unique entities
        df['all_entities'] = df.apply(
            lambda row: self._combine_entities(
                row['title_entities'], 
                row['abstract_entities']
            ), 
            axis=1
        )
        
        logger.info(f"Loaded {len(df)} news articles")
        logger.info(f"Categories: {df['category'].nunique()}")
        logger.info(f"Subcategories: {df['subcategory'].nunique()}")
        
        return df
    
    def _parse_entities(self, entities_str: str) -> List[Dict]:
        """Parse entity JSON string."""
        try:
            entities = json.loads(entities_str)
            return entities if isinstance(entities, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    
    def _combine_entities(self, title_entities: List[Dict], 
                          abstract_entities: List[Dict]) -> List[str]:
        """
        Combine and deduplicate entities from title and abstract.
        
        Returns:
            List of unique WikidataIds
        """
        entity_ids = set()
        
        for entity in title_entities + abstract_entities:
            if 'WikidataId' in entity:
                entity_ids.add(entity['WikidataId'])
        
        return list(entity_ids)
    
    def load_entity_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load entity embeddings from entity_embedding.vec.
        
        Returns:
            Dictionary mapping entity_id to embedding vector
        """
        embeddings_path = self.data_dir / 'entity_embedding.vec'
        
        if not embeddings_path.exists():
            logger.warning("entity_embedding.vec not found!")
            return {}
        
        logger.info(f"Loading entity embeddings from {embeddings_path}...")
        
        embeddings = {}
        with open(embeddings_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    entity_id = parts[0]
                    vector = np.array([float(x) for x in parts[1:]])
                    embeddings[entity_id] = vector
        
        logger.info(f"Loaded {len(embeddings)} entity embeddings")
        return embeddings
    
    def load_relation_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load relation embeddings from relation_embedding.vec.
        
        Returns:
            Dictionary mapping relation_id to embedding vector
        """
        embeddings_path = self.data_dir / 'relation_embedding.vec'
        
        if not embeddings_path.exists():
            logger.warning("relation_embedding.vec not found!")
            return {}
        
        logger.info(f"Loading relation embeddings from {embeddings_path}...")
        
        embeddings = {}
        with open(embeddings_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    relation_id = parts[0]
                    vector = np.array([float(x) for x in parts[1:]])
                    embeddings[relation_id] = vector
        
        logger.info(f"Loaded {len(embeddings)} relation embeddings")
        return embeddings


class MINDPreprocessor:
    """
    Preprocessor for MIND dataset to prepare features for recommendation
    with diversity considerations.
    """
    
    def __init__(self, loader: MINDDataLoader):
        """
        Initialize preprocessor with loaded data.
        
        Args:
            loader: MINDDataLoader instance with loaded data
        """
        self.loader = loader
        self.news_df = loader.news_df.copy()
        self.behaviors_df = loader.behaviors_df.copy()
        
        # Encodings
        self.category_encoder = None
        self.subcategory_encoder = None
        self.news_id_to_idx = None
        self.user_id_to_idx = None
        
        # Statistics
        self.category_distribution = None
        self.provider_distribution = None  # If provider info available
        
    def preprocess_all(self) -> Dict:
        """
        Run all preprocessing steps and return processed data.
        
        Returns:
            Dictionary containing processed datasets and encoders
        """
        logger.info("Starting preprocessing pipeline...")
        
        # 1. Clean and encode categorical features
        self.encode_categories()
        
        # 2. Create mappings
        self.create_id_mappings()
        
        # 3. Build user interaction matrix
        user_item_matrix = self.build_user_item_matrix()
        
        # 4. Process news features
        news_features = self.extract_news_features()
        
        # 5. Calculate diversity metrics base stats
        diversity_stats = self.calculate_diversity_statistics()
        
        # 6. Prepare interaction data
        interactions = self.prepare_interaction_data()

        logger.info("Preprocessing complete!")

        return {
            'news_features': news_features,
            'user_item_matrix': user_item_matrix,
            'interactions': interactions,
            'diversity_stats': diversity_stats,
            'encoders': {
                'category': self.category_encoder,
                'subcategory': self.subcategory_encoder,
                'news_id_to_idx': self.news_id_to_idx,
                'user_id_to_idx': self.user_id_to_idx
            },
            'news_df': self.news_df,
            'behaviors_df': self.behaviors_df
        }
    
    def encode_categories(self):
        """Encode categorical features (category, subcategory)."""
        logger.info("Encoding categorical features...")
        
        # Category encoding
        self.category_encoder = {
            cat: idx for idx, cat in enumerate(self.news_df['category'].unique())
        }
        self.news_df['category_encoded'] = self.news_df['category'].map(
            self.category_encoder
        )
        
        # Subcategory encoding
        self.subcategory_encoder = {
            subcat: idx for idx, subcat in enumerate(
                self.news_df['subcategory'].unique()
            )
        }
        self.news_df['subcategory_encoded'] = self.news_df['subcategory'].map(
            self.subcategory_encoder
        )
        
        logger.info(f"Encoded {len(self.category_encoder)} categories")
        logger.info(f"Encoded {len(self.subcategory_encoder)} subcategories")
    
    def create_id_mappings(self):
        """Create mappings from IDs to indices."""
        logger.info("Creating ID mappings...")
        
        # News ID to index
        self.news_id_to_idx = {
            news_id: idx for idx, news_id in enumerate(
                self.news_df['news_id'].unique()
            )
        }
        
        # User ID to index
        self.user_id_to_idx = {
            user_id: idx for idx, user_id in enumerate(
                self.behaviors_df['user_id'].unique()
            )
        }
        
        logger.info(f"Created mappings for {len(self.news_id_to_idx)} news articles")
        logger.info(f"Created mappings for {len(self.user_id_to_idx)} users")
    
    def build_user_item_matrix(self) -> Dict:
        """
        Build user-item interaction matrix from behaviors.
        
        Returns:
            Dictionary with user interaction data
        """
        logger.info("Building user-item interaction matrix...")
        
        user_histories = defaultdict(list)
        user_impressions = defaultdict(list)
        
        for _, row in self.behaviors_df.iterrows():
            user_id = row['user_id']
            
            # Store history
            if row['history']:
                user_histories[user_id].extend(row['history'])
            
            # Store impressions with labels
            if row['impressions']:
                user_impressions[user_id].extend(row['impressions'])
        
        # Deduplicate histories while preserving order
        for user_id in user_histories:
            seen = set()
            deduplicated = []
            for news_id in user_histories[user_id]:
                if news_id not in seen:
                    seen.add(news_id)
                    deduplicated.append(news_id)
            user_histories[user_id] = deduplicated
        
        logger.info(f"Built interaction data for {len(user_histories)} users")
        
        return {
            'user_histories': dict(user_histories),
            'user_impressions': dict(user_impressions)
        }
    
    def extract_news_features(self) -> pd.DataFrame:
        """
        Extract and organize news features for model input.
        
        Returns:
            DataFrame with processed news features
        """
        logger.info("Extracting news features...")
        
        features_df = self.news_df[[
            'news_id', 'category', 'subcategory', 'category_encoded',
            'subcategory_encoded', 'title', 'abstract', 'combined_text',
            'all_entities'
        ]].copy()
        
        # Add entity count
        features_df['entity_count'] = features_df['all_entities'].apply(len)
        
        # Add text length features
        features_df['title_length'] = features_df['title'].apply(
            lambda x: len(str(x).split())
        )
        features_df['abstract_length'] = features_df['abstract'].apply(
            lambda x: len(str(x).split())
        )
        
        # Add entity embeddings (average if multiple entities)
        if self.loader.entity_embeddings:
            features_df['entity_embedding'] = features_df['all_entities'].apply(
                lambda entities: self._get_averaged_entity_embedding(entities)
            )
        
        logger.info(f"Extracted features for {len(features_df)} news articles")
        
        return features_df
    
    def _get_averaged_entity_embedding(self, entity_ids: List[str]) -> Optional[np.ndarray]:
        """
        Get averaged embedding for a list of entities.
        
        Args:
            entity_ids: List of WikidataIds
        
        Returns:
            Averaged embedding vector or None
        """
        if not entity_ids:
            return None
        
        embeddings = []
        for entity_id in entity_ids:
            if entity_id in self.loader.entity_embeddings:
                embeddings.append(self.loader.entity_embeddings[entity_id])
        
        if not embeddings:
            return None
        
        return np.mean(embeddings, axis=0)
    
    def calculate_diversity_statistics(self) -> Dict:
        """
        Calculate statistics needed for diversity-aware ranking.
        
        Returns:
            Dictionary with diversity-related statistics
        """
        logger.info("Calculating diversity statistics...")
        
        # Category distribution
        category_dist = self.news_df['category'].value_counts(normalize=True).to_dict()
        
        # Subcategory distribution
        subcategory_dist = self.news_df['subcategory'].value_counts(
            normalize=True
        ).to_dict()
        
        # Entity diversity (unique entities per category)
        entity_diversity = {}
        for category in self.news_df['category'].unique():
            cat_news = self.news_df[self.news_df['category'] == category]
            all_entities = set()
            for entities in cat_news['all_entities']:
                all_entities.update(entities)
            entity_diversity[category] = len(all_entities)
        
        stats = {
            'category_distribution': category_dist,
            'subcategory_distribution': subcategory_dist,
            'entity_diversity_by_category': entity_diversity,
            'total_categories': len(category_dist),
            'total_subcategories': len(subcategory_dist)
        }
        
        logger.info(f"Diversity stats: {stats['total_categories']} categories, "
                   f"{stats['total_subcategories']} subcategories")
        
        return stats
    
    def _calculate_user_category_diversity(self) -> Dict:
        """
        Calculate category diversity in user histories.
        
        Returns:
            Dictionary mapping user_id to category diversity metrics
        """
        user_diversity = {}
        
        news_category_map = dict(zip(
            self.news_df['news_id'], 
            self.news_df['category']
        ))
        
        for _, row in self.behaviors_df.iterrows():
            user_id = row['user_id']
            history = row['history']
            
            if not history:
                continue
            
            categories = [
                news_category_map.get(news_id) 
                for news_id in history 
                if news_id in news_category_map
            ]
            
            if categories:
                unique_categories = len(set(categories))
                total_items = len(categories)
                user_diversity[user_id] = {
                    'unique_categories': unique_categories,
                    'total_items': total_items,
                    'diversity_ratio': unique_categories / total_items
                }
        
        return user_diversity
    
    def prepare_interaction_data(self) -> List:
        """
        Prepare interaction data from behaviors.

        The MIND dataset provides official train (MINDlarge_train) and
        validation (MINDlarge_dev) splits, so no further splitting is done here.

        Returns:
            List of interaction dicts for the loaded dataset split.
        """
        logger.info("Preparing interaction data...")

        interactions = []

        for _, row in self.behaviors_df.iterrows():
            user_id = row['user_id']
            user_idx = self.user_id_to_idx.get(user_id)

            if user_idx is None:
                continue

            history = row['history']
            history_indices = [
                self.news_id_to_idx[nid]
                for nid in history
                if nid in self.news_id_to_idx
            ]

            for news_id, label in row['impressions']:
                if news_id in self.news_id_to_idx:
                    news_idx = self.news_id_to_idx[news_id]

                    interactions.append({
                        'user_id': user_id,
                        'user_idx': user_idx,
                        'news_id': news_id,
                        'news_idx': news_idx,
                        'label': label,
                        'history': history,
                        'history_indices': history_indices,
                        'impression_id': row['impression_id'],
                        'time': row['time']
                    })

        logger.info(f"Prepared {len(interactions)} interaction samples")
        return interactions


def get_data_summary(loader: MINDDataLoader, preprocessor: MINDPreprocessor) -> Dict:
    """
    Generate comprehensive summary of the loaded and processed data.
    
    Args:
        loader: MINDDataLoader instance
        preprocessor: MINDPreprocessor instance
    
    Returns:
        Dictionary with data summary statistics
    """
    summary = {
        'dataset_type': loader.dataset_type,
        'data_path': str(loader.data_dir),
        
        # Behaviors
        'total_impressions': len(loader.behaviors_df),
        'unique_users': loader.behaviors_df['user_id'].nunique(),
        'avg_history_length': loader.behaviors_df['history'].apply(len).mean(),
        'avg_impressions_per_user': loader.behaviors_df.groupby('user_id').size().mean(),
        
        # News
        'total_news': len(loader.news_df),
        'categories': loader.news_df['category'].nunique(),
        'subcategories': loader.news_df['subcategory'].nunique(),
        'news_with_entities': (loader.news_df['all_entities'].apply(len) > 0).sum(),
        
        # Embeddings
        'entity_embeddings_available': len(loader.entity_embeddings) > 0,
        'num_entity_embeddings': len(loader.entity_embeddings),
        'num_relation_embeddings': len(loader.relation_embeddings),
        
        # Category breakdown
        'category_distribution': loader.news_df['category'].value_counts().to_dict(),
        
        # Temporal range
        'time_range': {
            'start': loader.behaviors_df['time'].min(),
            'end': loader.behaviors_df['time'].max(),
            'duration_days': (
                loader.behaviors_df['time'].max() - 
                loader.behaviors_df['time'].min()
            ).days
        }
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("MIND Dataset Loader - Example Usage")
    print("=" * 60)
    
    # Initialize loader
    data_dir = "./MINDlarge_train"  # Change to your data directory
    loader = MINDDataLoader(data_dir, dataset_type='train')
    
    # Load all data
    loader.load_all_data()
    
    # Initialize preprocessor
    preprocessor = MINDPreprocessor(loader)
    
    # Run preprocessing
    processed_data = preprocessor.preprocess_all()
    
    # Get summary
    summary = get_data_summary(loader, preprocessor)
    
    print("\nData Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    print("\nProcessed data keys:")
    print(list(processed_data.keys()))
