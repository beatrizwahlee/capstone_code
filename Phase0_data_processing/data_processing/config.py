"""
Configuration file for MIND dataset processing and modeling.

Adjust these parameters based on your needs and computational resources.
"""

import os
from pathlib import Path


class Config:
    """Configuration class for the news recommendation system."""
    
    # =========================================================================
    # DATA PATHS
    # =========================================================================
    
    # Root directory for all data
    DATA_ROOT = Path("./data")
    
    # Raw data directories
    TRAIN_DATA_DIR = DATA_ROOT / "raw" / "MINDlarge_train"
    VALID_DATA_DIR = DATA_ROOT / "raw" / "MINDlarge_dev"
    
    # Processed data directory
    PROCESSED_DATA_DIR = DATA_ROOT / "processed"
    
    # Output directories
    OUTPUT_DIR = Path("./outputs")
    MODEL_DIR = OUTPUT_DIR / "models"
    RESULTS_DIR = OUTPUT_DIR / "results"
    LOGS_DIR = OUTPUT_DIR / "logs"
    
    # =========================================================================
    # DATA PROCESSING
    # =========================================================================
    
    # Dataset type: 'small' or 'large'
    DATASET_SIZE = 'large'  # Change to 'small' for faster testing
    
    # Whether to extract from zip files
    EXTRACT_FROM_ZIP = False
    TRAIN_ZIP_PATH = None  # Set if using zip
    VALID_ZIP_PATH = None  # Set if using zip
    
    # Maximum number of items in user history to use
    MAX_HISTORY_LENGTH = 50
    
    # Minimum number of clicks for a user to be included
    MIN_USER_CLICKS = 1
    
    # Train/validation split ratio (if splitting training data)
    TRAIN_VAL_SPLIT = 0.8
    
    # =========================================================================
    # TEXT PROCESSING
    # =========================================================================
    
    # Maximum text length (in words)
    MAX_TITLE_LENGTH = 30
    MAX_ABSTRACT_LENGTH = 100
    
    # Text preprocessing options
    LOWERCASE_TEXT = True
    REMOVE_STOPWORDS = False  # Set to True if needed
    STEMMING = False  # Set to True if needed
    
    # =========================================================================
    # EMBEDDINGS
    # =========================================================================
    
    # Embedding dimensions
    ENTITY_EMBEDDING_DIM = 100  # Fixed by MIND dataset
    RELATION_EMBEDDING_DIM = 100  # Fixed by MIND dataset
    
    # Text embedding model (for future use)
    TEXT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    TEXT_EMBEDDING_DIM = 384  # Depends on model chosen
    
    # User embedding dimension (for neural model)
    USER_EMBEDDING_DIM = 128
    
    # News embedding dimension (for neural model)
    NEWS_EMBEDDING_DIM = 256
    
    # Whether to use pretrained entity embeddings
    USE_ENTITY_EMBEDDINGS = True
    
    # =========================================================================
    # MODEL ARCHITECTURE (for future relevance model)
    # =========================================================================
    
    # Neural network architecture
    HIDDEN_DIMS = [256, 128, 64]  # Hidden layer dimensions
    DROPOUT_RATE = 0.2
    ACTIVATION = 'relu'  # 'relu', 'gelu', 'tanh'
    
    # Attention mechanism
    USE_ATTENTION = True
    ATTENTION_HEADS = 4
    ATTENTION_DIM = 64
    
    # =========================================================================
    # TRAINING (for future use)
    # =========================================================================
    
    # Training hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 3
    
    # Optimizer
    OPTIMIZER = 'adam'  # 'adam', 'sgd', 'adamw'
    WEIGHT_DECAY = 1e-5
    
    # Loss function
    LOSS_FUNCTION = 'binary_crossentropy'  # For click prediction
    
    # Device
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    
    # =========================================================================
    # DIVERSITY & RE-RANKING
    # =========================================================================
    
    # Re-ranking parameters
    RERANK_SIZE = 20  # Number of top candidates to re-rank
    
    # Diversity weight (0 = pure relevance, 1 = pure diversity)
    DIVERSITY_WEIGHT = 0.3
    
    # Category diversity weight
    CATEGORY_DIVERSITY_WEIGHT = 0.4
    
    # Calibration weight (match user's historical distribution)
    CALIBRATION_WEIGHT = 0.3
    
    # Provider fairness weight
    PROVIDER_FAIRNESS_WEIGHT = 0.2
    
    # Novelty weight (promote less popular items)
    NOVELTY_WEIGHT = 0.1
    
    # Re-ranking algorithm
    RERANK_ALGORITHM = 'mmr'  # 'mmr', 'xquad', 'pm2', 'greedy'
    
    # MMR lambda parameter (relevance vs diversity tradeoff)
    MMR_LAMBDA = 0.5
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    
    # Metrics to compute
    METRICS = [
        'auc',
        'mrr',
        'ndcg@5',
        'ndcg@10',
        'diversity@10',
        'coverage@10',
        'calibration@10',
        'gini@10'
    ]
    
    # Top-K values for ranking metrics
    TOP_K_VALUES = [5, 10, 20]
    
    # =========================================================================
    # DIVERSITY METRICS
    # =========================================================================
    
    # Diversity calculation method
    DIVERSITY_METRIC = 'ild'  # 'ild', 'entropy', 'coverage'
    
    # Similarity measure for ILD
    SIMILARITY_MEASURE = 'cosine'  # 'cosine', 'jaccard'
    
    # Category diversity calculation
    CATEGORY_DIVERSITY_METHOD = 'entropy'  # 'entropy', 'count', 'gini'
    
    # =========================================================================
    # LOGGING & DEBUGGING
    # =========================================================================
    
    # Logging level
    LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    
    # Save intermediate results
    SAVE_INTERMEDIATE = True
    
    # Verbose output
    VERBOSE = True
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # =========================================================================
    # COMPUTATIONAL RESOURCES
    # =========================================================================
    
    # Number of CPU workers for data loading
    NUM_WORKERS = 4
    
    # Use mixed precision training (for GPU)
    USE_MIXED_PRECISION = False
    
    # Gradient accumulation steps
    GRADIENT_ACCUMULATION_STEPS = 1
    
    # =========================================================================
    # EXPERIMENT TRACKING (for future use)
    # =========================================================================
    
    # Experiment name
    EXPERIMENT_NAME = 'diversity_aware_recommender'
    
    # Whether to use MLflow/Weights&Biases for tracking
    USE_EXPERIMENT_TRACKING = False
    TRACKING_BACKEND = 'mlflow'  # 'mlflow', 'wandb', 'tensorboard'
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories."""
        dirs = [
            cls.DATA_ROOT,
            cls.PROCESSED_DATA_DIR,
            cls.OUTPUT_DIR,
            cls.MODEL_DIR,
            cls.RESULTS_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✓ All directories created")
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings."""
        issues = []
        
        # Check if data directories exist
        if not cls.TRAIN_DATA_DIR.exists() and not cls.EXTRACT_FROM_ZIP:
            issues.append(f"Training data directory not found: {cls.TRAIN_DATA_DIR}")
        
        if not cls.VALID_DATA_DIR.exists() and not cls.EXTRACT_FROM_ZIP:
            issues.append(f"Validation data directory not found: {cls.VALID_DATA_DIR}")
        
        # Check parameter ranges
        if not 0 <= cls.DIVERSITY_WEIGHT <= 1:
            issues.append("DIVERSITY_WEIGHT must be between 0 and 1")
        
        if not 0 <= cls.MMR_LAMBDA <= 1:
            issues.append("MMR_LAMBDA must be between 0 and 1")
        
        if cls.BATCH_SIZE <= 0:
            issues.append("BATCH_SIZE must be positive")
        
        if cls.LEARNING_RATE <= 0:
            issues.append("LEARNING_RATE must be positive")
        
        if issues:
            print("⚠️  Configuration Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✓ Configuration validated successfully")
            return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "=" * 80)
        print("CONFIGURATION SETTINGS")
        print("=" * 80)
        
        print("\n--- Data Paths ---")
        print(f"Train data: {cls.TRAIN_DATA_DIR}")
        print(f"Valid data: {cls.VALID_DATA_DIR}")
        print(f"Processed data: {cls.PROCESSED_DATA_DIR}")
        
        print("\n--- Model Settings ---")
        print(f"User embedding dim: {cls.USER_EMBEDDING_DIM}")
        print(f"News embedding dim: {cls.NEWS_EMBEDDING_DIM}")
        print(f"Hidden layers: {cls.HIDDEN_DIMS}")
        print(f"Dropout: {cls.DROPOUT_RATE}")
        
        print("\n--- Training Settings ---")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Device: {cls.DEVICE}")
        
        print("\n--- Diversity Settings ---")
        print(f"Diversity weight: {cls.DIVERSITY_WEIGHT}")
        print(f"Category diversity: {cls.CATEGORY_DIVERSITY_WEIGHT}")
        print(f"Calibration weight: {cls.CALIBRATION_WEIGHT}")
        print(f"Re-ranking algorithm: {cls.RERANK_ALGORITHM}")
        
        print("\n--- Evaluation Settings ---")
        print(f"Metrics: {', '.join(cls.METRICS)}")
        print(f"Top-K values: {cls.TOP_K_VALUES}")
        
        print("\n" + "=" * 80)
    
    @classmethod
    def save_config(cls, filepath: str = None):
        """Save configuration to JSON file."""
        import json
        
        if filepath is None:
            filepath = cls.OUTPUT_DIR / 'config.json'
        
        config_dict = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(cls).items()
            if not key.startswith('_') and not callable(value)
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ Configuration saved to {filepath}")


# Example of creating different configs for different experiments
class SmallDatasetConfig(Config):
    """Configuration for quick testing with small dataset."""
    DATASET_SIZE = 'small'
    TRAIN_DATA_DIR = Config.DATA_ROOT / "raw" / "MINDsmall_train"
    VALID_DATA_DIR = Config.DATA_ROOT / "raw" / "MINDsmall_dev"
    BATCH_SIZE = 32
    NUM_EPOCHS = 5


class HighDiversityConfig(Config):
    """Configuration emphasizing diversity over relevance."""
    DIVERSITY_WEIGHT = 0.5
    CATEGORY_DIVERSITY_WEIGHT = 0.6
    CALIBRATION_WEIGHT = 0.3
    PROVIDER_FAIRNESS_WEIGHT = 0.1
    MMR_LAMBDA = 0.3  # More diversity


class HighRelevanceConfig(Config):
    """Configuration emphasizing relevance over diversity."""
    DIVERSITY_WEIGHT = 0.1
    CATEGORY_DIVERSITY_WEIGHT = 0.2
    CALIBRATION_WEIGHT = 0.1
    PROVIDER_FAIRNESS_WEIGHT = 0.05
    MMR_LAMBDA = 0.8  # More relevance


if __name__ == "__main__":
    # Example usage
    print("Default Configuration:")
    Config.create_directories()
    Config.print_config()
    Config.validate_config()
    Config.save_config()
    
    print("\n" + "=" * 80)
    print("Alternative configurations available:")
    print("  - SmallDatasetConfig: For quick testing")
    print("  - HighDiversityConfig: Emphasizes diversity")
    print("  - HighRelevanceConfig: Emphasizes relevance")
