"""
main script for the project
"""

import argparse
import logging
import os
import time
import polars as pl
from pathlib import Path
import gc
gc.collect()  # Manually free memory
from sklearn.model_selection import train_test_split

# local imports
import config
from data_loader import check_data_exists
from data import PreprocessData
from feat_selection import FeatureSelection
from utils import reduce_mem_usage, filter_cols, set_table_dtypes, reduce_mem_usage_pl_custom
from model import ModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[
        logging.FileHandler(config.ROOT_DIR / "logs" / "project.log"),])

logger = logging.getLogger(__name__)

# Set up project directories
def setup_directories():
    """
    Create directories for the project.
    """
    for directory in [config.DATA_DIR, config.DATA_TRAIN, config.DATA_TEST, config.FEATURES_DIR, 
                      config.MODELS_DIR, config.REPORTS_DIR, config.FIGURES_DIR, config.LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    logger.info("All directories created.")

def check_data():
    """
    Check data files.
    """
    logger.info("Loading data...")
    logger.info("Data availabilities:")

    # Check if data files exist
    availabilities = check_data_exists()
    misssing_files = [file for file, available in availabilities.items() if not available]

    if misssing_files:
        logger.warning("Missing files: {} data files are not available.".format(len(misssing_files)))
        logger.info("data directory: {}".format(config.DATA_DIR))
        logger.info("missing files:")
        for file in misssing_files:
            logger.info("{}".format(file))
        return False
    else:
        logger.info("All data files are available.")
        return True

def preprocess_data():
    """
    Preprocess data.
    """
    logger.info("Starting exploratory data analysis...")

    start_time = time.time()
    
    data = PreprocessData()
    feat_met = FeatureSelection()
    logger.info("Loading data...")

    data.load_data()
    
    logger.info("Preprocessing data...")

    data.drop_missing_value_data(mode='train')

    data.aggregate_data()

    logger.info("Joining data...")
    df_train = data.join_data()
    logger.info("Shape of joined data: {}".format(df_train.shape))

    end_time = time.time()
    duration = end_time - start_time
    logger.info("Preprocessing complete. Duration: {:.2f} minutes.".format(duration / 60))

    # feature selection
    start_time = time.time()
    logger.info("Feature selection...")
    methods = ['correlation', 'lasso', 'random_forest']
    logger.info(f"Runing feature select with technique: {methods[0]}, {methods[1]}, {methods[2]}")
    feats = feat_met.run_selection(df_train, methods=methods, y='target')
    end_time = time.time()
    duration = end_time - start_time
    logger.info("Total feature selection duration: {:.2f} minutes.".format(duration / 60))
    for feat in feats:
        logger.info(f"Features selected by: {feat.capitalize()}")
        logger.info(f"{'-' * 20} {feat.capitalize()} {'-' * 20}")
        logger.info(f"Number of features selected: {feats[feat].shape[1]}")
        logger.info(f"Features: {feats[feat].drop(config.KEY_COLUMNS).columns}")
        logger.info('\n')
    return feats

def train(data, feature_names):
    """
    Train the model.
    """
    logger.info("Preparing data for training...")
    X = data #data.drop(columns=config.KEY_COLUMNS,axis=1)
    y = data[config.KEY_COLUMNS[2]]
    logger.info("Training the model...")
    # Initialize model trainer
    trainer = ModelTrainer(X,y)
    results = trainer.train_models(['LGBM','XGBoost','CatBoost','Logistic Regression'],feature_names)
    print("\nModel Training Results:")
    print("=" * 80)
    for model, res in results.items():
        print(f"Model: {model}")
        print(f"{'-' * 80}")
        print(f"Accuracy: {res['accuracy']}")
        print(f"Precision: {res['precision']}")
        print(f"Recall: {res['recall']}")
        print(f"F1_score: {res['f1_score']}")
        print(f"AUC: {res['auc']}")
        print(f"Train Gini: {res['train_gini']}")
        print(f"Val Gini: {res['val_gini']}")
        print(f"Train Gini Stability: {res['train_gini_stability']}")
        print(f"Val Gini Stability: {res['val_gini_stability']}")
        print('\n')
    print("=" * 80)

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the project.")
    parser.add_argument("--setup", type=str, choices=['train', 'eval', 'test'], 
                        default='train', help="Pipeline step to run")
    return parser.parse_args()

if __name__ == "__main__":
    logger.info("Script started.")

    # Parse command line arguments
    args = parse_args()
    
    # check_data()
    # final_data = preprocess_data()

    # for key,data in final_data.items():
    #     logger.info(f"Training with features selected by: {key.capitalize()}")
    #     print(f"\n{'-' * 20} {key.capitalize()} {'-' * 20}")
    #     train(data.to_pandas(),key)

    if args.setup == 'train':
        # check_data()
        if not check_data():
            logger.error("Data files are missing. Exiting script.")
        final_data = preprocess_data()
        # print("shape of final data: {}".format(final_data.shape))
        for key,data in final_data.items():
            logger.info(f"Training with features selected by: {key.capitalize()}")
            print(f"\n{'-' * 20} {key.capitalize()} {'-' * 20}")
            train(data.to_pandas(),key)
    else:
        pass    
        # Load data

    logger.info("Project setup complete.")