"""
Utility functions for loading data from the data folder.
"""

import pandas as pd
import polars as pl
import numpy as np
import pyarrow.parquet as pq
import logging
import os
from typing import List, Tuple, Dict
import gc
gc.collect()  # Manually free memory

# import local modules
import config
from utils import reduce_mem_usage, set_table_dtypes, reduce_mem_usage_pl, reduce_mem_usage_pl_custom
from pathlib import Path


# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)

def load_base_tables(mode: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load the base tables.
    :return: Tuple of train and test base tables.
    """

    logger.info("Loading base tables...")
    df = None
    file = None
    if mode == "train":
        file = Path(config.TRAIN_BASE)
    elif mode == "test":
        file = Path(config.TEST_BASE)

    try:
        df = pl.read_parquet(file).pipe(set_table_dtypes)
        df = pl.DataFrame(reduce_mem_usage_pl_custom(df))
        logger.info("Loaded {} base table. with shape: {}".format(mode,df.shape))
    except FileNotFoundError:
        logger.error("{} base table not found.",format(mode))
        df = pl.DataFrame()

    return df

def load_feature_definitions():
    """
    Load the feature definitions.
    :return: DataFrame of feature definitions.
    """

    logger.info("Loading feature definitions...")

    try:
        feature_definitions = pl.read_csv(config.FEATURE_DEFINITIONS)
        logger.info("Loaded feature definitions with shape: {}".format(feature_definitions.shape))
    except FileNotFoundError:
        logger.error("Feature definitions file not found.")
        feature_definitions = pl.DataFrame()

    return feature_definitions
# def read_and_filter_table(file_path):
#     """
#     Read a table and filter columns based on similar variables analysis.
#     Keep only the recommended variables (Variable_To_Keep) from similar variables analysis.
    
#     Parameters:
#     -----------
#     file_path : str or Path
#         Path to the data file (parquet or csv)
        
#     Returns:
#     --------
#     pl.DataFrame
#         Filtered DataFrame with only recommended variables
#     """
#     # Read the similar variables analysis
#     similar_vars_df = pd.read_csv(config.SIMILAR_VARIABLES_ANALYSIS)
#     # Get the list of variables to keep
#     vars_to_keep = similar_vars_df['Variable_To_Keep'].tolist()
#     # Determine file extension
#     file_extension = os.path.splitext(file_path)[1].lower()
    
#     try:
#         # Read the data file
#         if file_extension == '.parquet':
#             df = pl.read_parquet(file_path)
#         elif file_extension in ['.csv', '.txt']:
#             df = pl.read_csv(file_path)
#         else:
#             raise ValueError(f"Unsupported file extension: {file_extension}")
        
#         # Get current columns
#         current_columns = df.columns
        
#         # Add required columns like case_id, target, etc. that should always be kept
#         required_columns = config.KEY_COLUMNS
#         keep_columns = [col for col in required_columns if col in current_columns]
        
#         # Find which variables to keep in the current table
#         keep_vars = [var for var in vars_to_keep if var in current_columns]
        
#         # Add these to the keep_columns list
#         return keep_columns.extend(keep_vars)
    
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         return None
    
def load_table(table_name: str, files: List[str], mode: str, prefix: str) -> pl.DataFrame:
    """
    Load a table from the data folder.
    :param table_name: Name of the table to load.
    :param files: List of parquet file paths to be merged.
    :param mode: Mode of data to load. Choose from 'train' or 'test'.
    :param prefix: Prefix of the files to be merged.
    """

    logger.info("Loading table: {} {}...".format(mode, table_name))
    
    data_path = None
    table = None

    if mode == "train":
        data_path = config.DATA_TRAIN
    elif mode == "test":
        data_path = config.DATA_TEST
    
    try:
        table = join_same_depth_data(table_name, files, data_path, mode, prefix)
        logger.info("Loaded table {} {} with shape: {}".format(mode, table_name, table.shape))
    except FileNotFoundError:
        logger.error("Table not found.")
        table = pl.DataFrame()

    return table

def join_same_depth_data(table_name: str, files: List[str], data_path:str, mode: str, prefix: str) -> pl.DataFrame:
    """
    Join multiple parquet files that belong to the same depth and have the same features.
    :param table_name: Name of the table to load.
    :param files: List of parquet file paths to be merged.
    :param data_path: Path to the data folder.
    :param mode: Mode of data to load. Choose from 'train' or 'test'.
    :param prefix: Prefix of the files to be merged.
    :return: Merged DataFrame.
    """

    df_list = [os.path.join(data_path,f"{mode}_{file}.parquet") for file in files if prefix in file]
    
    if len(df_list) == 0:
        logger.error("No files found.")
        return pl.DataFrame()

    dfs = []
    # if DOMAIN_KNOWLEDGE:
    #     dfs = [reduce_mem_usage_pl_custom(pl.read_parquet(file).select(read_and_filter_table(file)).pipe(set_table_dtypes)) for file in df_list]
    #     # if table_name == "train_static_0":
    #     #     dfs = [reduce_mem_usage_pl_custom(pl.read_parquet(file).select(config.STATIC_0_USES).pipe(set_table_dtypes)) for file in df_list]
    #     # elif table_name == "train_credit_bureau_a_1":
    #     #     dfs = [reduce_mem_usage_pl_custom(pl.read_parquet(file).select(config.CREDIT_BUREAU_A_1_USES).pipe(set_table_dtypes)) for file in df_list]
    #     # elif table_name == "train_credit_bureau_a_2":
    #     #     dfs = [reduce_mem_usage_pl_custom(pl.read_parquet(file).select(config.CREDIT_BUREAU_A_2_USES).pipe(set_table_dtypes)) for file in df_list]
    #     # elif table_name == "train_credit_bureau_b_1":
    #     #     dfs = [reduce_mem_usage_pl_custom(pl.read_parquet(file).select(config.CREDIT_BUREAU_B_1_USES).pipe(set_table_dtypes)) for file in df_list]
    #     # else:
    #     #     dfs = [reduce_mem_usage_pl_custom(pl.read_parquet(file).pipe(set_table_dtypes)) for file in df_list]
    # else:
    dfs = [reduce_mem_usage_pl_custom(pl.read_parquet(file).pipe(set_table_dtypes)) for file in df_list]
    gc.collect()
    merged_df = pl.concat(dfs, how="vertical_relaxed")
    gc.collect()
    
    return merged_df

def check_data_exists():
    """
    Check if the data files exist.
    """

    availabilities = {}

    # Check base tables
    availabilities["train_base.parquet"] = os.path.exists(config.TRAIN_BASE)
    availabilities["test_base.parquet"] = os.path.exists(config.TEST_BASE)
    availabilities["feature_definitions.csv"] = os.path.exists(config.FEATURE_DEFINITIONS)

    # check depth 0 tables
    for table in config.DEPTH_0_TABLES_TRAIN:
        availabilities[f"train_{table}.parquet"] = os.path.exists(config.DATA_TRAIN / f"train_{table}.parquet")
    for table in config.DEPTH_0_TABLES_TEST:
        availabilities[f"test_{table}.parquet"] = os.path.exists(config.DATA_TEST / f"test_{table}.parquet")

    # check depth 1 tables
    for table in config.DEPTH_1_TABLES_TRAIN:
        availabilities[f"train_{table}.parquet"] = os.path.exists(config.DATA_TRAIN / f"train_{table}.parquet")
    for table in config.DEPTH_1_TABLES_TEST:
        availabilities[f"test_{table}.parquet"] = os.path.exists(config.DATA_TEST / f"test_{table}.parquet")

    # check depth 2 tables
    for table in config.DEPTH_2_TABLES_TRAIN:
        availabilities[f"train_{table}.parquet"] = os.path.exists(config.DATA_TRAIN / f"train_{table}.parquet")
    for table in config.DEPTH_2_TABLES_TEST:
        availabilities[f"test_{table}.parquet"] = os.path.exists(config.DATA_TEST / f"test_{table}.parquet")

    return availabilities

if __name__ == "__main__":
    # Load the data
    availabilities = check_data_exists()
    print ("Data availabilities:")
    for file, available in availabilities.items():
        noti_status = "Available" if available else "Not available"
        print("{}: ".format(file, noti_status))