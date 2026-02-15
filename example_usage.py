"""
Example Usage Script for MIND Dataset Loading and Preprocessing

This script demonstrates how to use the MINDDataLoader and MINDPreprocessor
classes to load, clean, and prepare the MIND dataset for your diversity-aware
news recommender system.

Run this script to get started with your capstone project!
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Import our custom modules
from mind_data_loader import MINDDataLoader, MINDPreprocessor, get_data_summary
from data_quality_checker import DataQualityChecker, print_dataset_statistics


def main():
    """Main execution function."""
    
    print("\n" + "=" * 80)
    print("MIND DATASET LOADING AND PREPROCESSING")
    print("Diversity-Aware News Recommender System - Capstone Project")
    print("=" * 80)
    
    # =========================================================================
    # STEP 1: Configure paths
    # =========================================================================
    print("\n[STEP 1] Configuring data paths...")
    
    # UPDATE THESE PATHS to match your data location
    TRAIN_DATA_DIR = "./MINDlarge_train"  
    VALID_DATA_DIR = "./MINDlarge_dev"   
    
    # Check if using zip files
    TRAIN_ZIP = None  # Set to "./MINDlarge_train.zip" if using zip
    VALID_ZIP = None  # Set to "./MINDlarge_dev.zip" if using zip
    
    print(f"Train data: {TRAIN_DATA_DIR}")
    print(f"Valid data: {VALID_DATA_DIR}")
    
    # =========================================================================
    # STEP 2: Load training data
    # =========================================================================
    print("\n[STEP 2] Loading training data...")
    
    train_loader = MINDDataLoader(TRAIN_DATA_DIR, dataset_type='train')
    
    # If you have zip files, extract them first
    if TRAIN_ZIP:
        train_loader.load_all_data(extract_zip=True, zip_path=TRAIN_ZIP)
    else:
        train_loader.load_all_data()
    
    print("✓ Training data loaded successfully!")
    
    # =========================================================================
    # STEP 3: Load validation data
    # =========================================================================
    print("\n[STEP 3] Loading validation data...")
    
    valid_loader = MINDDataLoader(VALID_DATA_DIR, dataset_type='valid')
    
    if VALID_ZIP:
        valid_loader.load_all_data(extract_zip=True, zip_path=VALID_ZIP)
    else:
        valid_loader.load_all_data()
    
    print("✓ Validation data loaded successfully!")
    
    # =========================================================================
    # STEP 4: Data quality checks
    # =========================================================================
    print("\n[STEP 4] Running data quality checks...")
    
    train_preprocessor = MINDPreprocessor(train_loader)
    
    checker = DataQualityChecker(train_loader, train_preprocessor)
    quality_report = checker.generate_report()
    
    print(quality_report)
    
    # Save quality report
    with open('data_quality_report.txt', 'w') as f:
        f.write(quality_report)
    print("✓ Quality report saved to 'data_quality_report.txt'")
    
    #hini amnhans
    # =========================================================================
    # STEP 5: Preprocess training data
    # =========================================================================
    print("\n[STEP 5] Preprocessing training data...")
    
    train_processed = train_preprocessor.preprocess_all()
    
    print("✓ Training data preprocessed!")
    print(f"  - News features: {len(train_processed['news_features'])} articles")
    print(f"  - Training samples: {len(train_processed['train_data'])}")
    print(f"  - Validation samples: {len(train_processed['val_data'])}")
    
    # =========================================================================
    # STEP 6: Preprocess validation data
    # =========================================================================
    print("\n[STEP 6] Preprocessing validation data...")
    
    valid_preprocessor = MINDPreprocessor(valid_loader)
    valid_processed = valid_preprocessor.preprocess_all()
    
    print("✓ Validation data preprocessed!")
    print(f"  - Test samples: {len(valid_processed['val_data'])}")
    
    # =========================================================================
    # STEP 7: Generate comprehensive statistics
    # =========================================================================
    print("\n[STEP 7] Generating dataset statistics...")
    
    print_dataset_statistics(train_loader, train_preprocessor)
    
    # =========================================================================
    # STEP 8: Save processed data
    # =========================================================================
    print("\n[STEP 8] Saving processed data...")
    
    # Create output directory
    output_dir = Path("./processed_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save news features
    train_processed['news_features'].to_csv(
        output_dir / 'news_features_train.csv', 
        index=False
    )
    print(f"✓ Saved news features: {output_dir / 'news_features_train.csv'}")
    
    # Save diversity statistics
    with open(output_dir / 'diversity_stats.json', 'w') as f:
        # Convert to JSON-serializable format
        diversity_stats = train_processed['diversity_stats'].copy()
        # Convert numpy values to Python types
        for key in diversity_stats:
            if isinstance(diversity_stats[key], dict):
                diversity_stats[key] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in diversity_stats[key].items()
                }
        json.dump(diversity_stats, f, indent=2)
    print(f"✓ Saved diversity stats: {output_dir / 'diversity_stats.json'}")
    
    # Save encoders
    encoders = train_processed['encoders']
    with open(output_dir / 'encoders.json', 'w') as f:
        json.dump({
            'category': encoders['category'],
            'subcategory': encoders['subcategory']
        }, f, indent=2)
    print(f"✓ Saved encoders: {output_dir / 'encoders.json'}")
    
    # Save interaction data (sample for inspection)
    sample_train = pd.DataFrame(train_processed['train_data'][:1000])
    sample_train.to_csv(output_dir / 'sample_train_interactions.csv', index=False)
    print(f"✓ Saved sample interactions: {output_dir / 'sample_train_interactions.csv'}")
    
    # =========================================================================
    # STEP 9: Generate data summary for your project report
    # =========================================================================
    print("\n[STEP 9] Generating comprehensive summary...")
    
    summary = get_data_summary(train_loader, train_preprocessor)
    
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total users: {summary['unique_users']:,}")
    print(f"Total news articles: {summary['total_news']:,}")
    print(f"Total impressions: {summary['total_impressions']:,}")
    print(f"Categories: {summary['categories']}")
    print(f"Subcategories: {summary['subcategories']}")
    print(f"Entity embeddings: {summary['num_entity_embeddings']:,}")
    print(f"Time range: {summary['time_range']['duration_days']} days")
    print("=" * 80)
    
    # =========================================================================
    # STEP 10: What's next?
    # =========================================================================
    print("\n[STEP 10] Next steps for your project:")
    print("-" * 80)
    print("1. ✓ Data loading and preprocessing complete!")
    print("2. → Text encoding: Implement NLP embeddings for news content")
    print("   - Consider using BERT, Sentence-BERT, or News-specific models")
    print("   - Encode titles and abstracts separately or combined")
    print("3. → User modeling: Build user interest profiles from history")
    print("   - Aggregate news embeddings from user history")
    print("   - Consider attention mechanisms for weighting")
    print("4. → Relevance model: Build click prediction model")
    print("   - Neural network with user and news embeddings")
    print("   - Train on click/no-click labels")
    print("5. → Diversity re-ranking: Implement multi-objective ranking")
    print("   - Category diversity (use category_distribution)")
    print("   - Provider diversity (if provider data available)")
    print("   - Viewpoint diversity (may need additional labeling)")
    print("   - Calibration (match user's historical category distribution)")
    print("6. → Evaluation: Implement metrics")
    print("   - Accuracy: AUC, MRR, nDCG")
    print("   - Diversity: ILD, coverage, entropy")
    print("   - Fairness: representation of minority categories")
    print("-" * 80)
    
    print("\n✓ All preprocessing complete! Data saved to './processed_data/'")
    print("\nYou can now proceed with building your recommendation model!")
    
    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'train_processed': train_processed,
        'valid_processed': valid_processed
    }


def quick_data_exploration(train_loader):
    """
    Quick exploration of the loaded data.
    Useful for understanding the data structure.
    """
    print("\n" + "=" * 80)
    print("QUICK DATA EXPLORATION")
    print("=" * 80)
    
    # Sample behavior
    print("\n--- Sample Behavior Record ---")
    sample_behavior = train_loader.behaviors_df.iloc[0]
    print(f"Impression ID: {sample_behavior['impression_id']}")
    print(f"User ID: {sample_behavior['user_id']}")
    print(f"Time: {sample_behavior['time']}")
    print(f"History length: {len(sample_behavior['history'])}")
    print(f"First 5 history items: {sample_behavior['history'][:5]}")
    print(f"Number of impressions: {len(sample_behavior['impressions'])}")
    print(f"First 3 impressions: {sample_behavior['impressions'][:3]}")
    
    # Sample news
    print("\n--- Sample News Article ---")
    sample_news = train_loader.news_df.iloc[0]
    print(f"News ID: {sample_news['news_id']}")
    print(f"Category: {sample_news['category']}")
    print(f"Subcategory: {sample_news['subcategory']}")
    print(f"Title: {sample_news['title']}")
    print(f"Abstract: {sample_news['abstract'][:100]}...")
    print(f"Number of entities: {len(sample_news['all_entities'])}")
    if sample_news['all_entities']:
        print(f"First entity: {sample_news['all_entities'][0]}")
    
    # Entity embedding
    if train_loader.entity_embeddings:
        sample_entity_id = list(train_loader.entity_embeddings.keys())[0]
        sample_embedding = train_loader.entity_embeddings[sample_entity_id]
        print(f"\n--- Sample Entity Embedding ---")
        print(f"Entity ID: {sample_entity_id}")
        print(f"Embedding dimension: {len(sample_embedding)}")
        print(f"First 10 values: {sample_embedding[:10]}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run the main pipeline
    try:
        results = main()
        
        # Optional: Run quick exploration
        print("\n" + "=" * 80)
        response = input("Would you like to see a quick data exploration? (y/n): ")
        if response.lower() == 'y':
            quick_data_exploration(results['train_loader'])
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✓ Script completed successfully!")
