"""
Data preprocessing module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import os
import polars as pl
from sklearn.preprocessing import LabelEncoder
import gc
gc.collect()  # Manually free memory

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# local imports
import config
from data_loader import load_base_tables, load_feature_definitions, load_table, join_same_depth_data
from utils import handle_dates, save_dataframe_as_parquet

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

class Aggregator:
    #Please add or subtract features yourself, be aware that too many features will take up too much space.
    def fill_nulls(df, excludes):
        """Fill numeric nulls with mean (skip non-numeric columns)"""
        # Safely get numeric columns (exclude non-numeric accidentally included)
        numeric_columns = [col for col in df.columns if df[col].dtype in pl.NUMERIC_DTYPES and col not in excludes]
        return df.with_columns([pl.col(col).fill_null(pl.mean(col)) for col in numeric_columns])
    
    def convert_categorical_to_numeric(df, excludes=[]):
        # Exclude 'case_id' from categorical columns
        categorical_columns = [col for col in df.columns if df[col].dtype not in pl.NUMERIC_DTYPES and col not in excludes]
        for col in categorical_columns:
            le = LabelEncoder()
            df = df.with_columns(pl.Series(col, le.fit_transform(df[col].to_list())))
        return df

    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_max = []
        for col in cols:
            expr_max.append(pl.mean(col).alias(f"avg_{col}"))
        return expr_max

    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return  expr_max
    
    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M") and df[col].dtype == pl.Utf8]
        expr_max = [pl.median(col).alias(f"median_{col}") for col in cols]
        return  expr_max

    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return  expr_max 
    
    # def count_expr(df):
    #     cols = [col for col in df.columns if "num_group" in col]
    #     expr_max = [pl.count().alias(f"count_{col}") for col in cols] 
    #     return  expr_max
    
    def get_exprs(df):
        exprs = Aggregator.num_expr(df) + \
                Aggregator.date_expr(df) + \
                Aggregator.str_expr(df) + \
                Aggregator.other_expr(df)
                # Aggregator.count_expr(df)
        return exprs
    
class PreprocessData:
    """
    Exploratory Data Analysis (EDA) class.
    """
    def __init__(self):
        """
        Initialize the class.
        """
        self.train_base = None
        self.test_base = None
        self.feature_definitions = None
        self.train_data = {}
        self.test_data = {}
        self.missing_threshold = config.MISSING_THRESHOLD
        self.agg = Aggregator()
        # Create reports directory if it doesn't exist
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    def load_data(self, mode="train"):
        """
        Load data files.
        """
        if mode == "train":
            logger.info("Loading train data...")
            self.feature_definitions = load_feature_definitions()
            # Load base tables
            self.train_base = load_base_tables(mode="train")
            self.train_data["train_base"] = self.train_base
            # from pdb import set_trace
            # set_trace()
            # Load tables of train data
            # Depth 0
            self.train_data["depth_0"] = {}
            self.train_data["depth_0"]["train_static_0"] = load_table(table_name="static_0_0", files=config.DEPTH_0_TABLES_TRAIN, mode="train", prefix="static_0")
            self.train_data["depth_0"]["train_static_cb_0"] = load_table(table_name="static_cb_0", files=config.DEPTH_0_TABLES_TRAIN, mode="train", prefix="static_cb")
            # Depth 1
            self.train_data["depth_1"] = {}
            self.train_data["depth_1"]["train_applprev_1"] = load_table(table_name="applprev_1_0", files=config.DEPTH_1_TABLES_TRAIN, mode="train", prefix="applprev_1")
            self.train_data["depth_1"]["train_other_1"] = load_table(table_name="other_1", files=config.DEPTH_1_TABLES_TRAIN, mode="train", prefix="other")
            self.train_data["depth_1"]["train_tax_registry_a_1"] = load_table(table_name="tax_registry_a_1", files=config.DEPTH_1_TABLES_TRAIN, mode="train", prefix="tax_registry_a")
            self.train_data["depth_1"]["train_tax_registry_b_1"] = load_table(table_name="tax_registry_b_1", files=config.DEPTH_1_TABLES_TRAIN, mode="train", prefix="tax_registry_b")
            self.train_data["depth_1"]["train_tax_registry_c_1"] = load_table(table_name="tax_registry_c_1", files=config.DEPTH_1_TABLES_TRAIN, mode="train", prefix="tax_registry_c")
            self.train_data["depth_1"]["train_credit_bureau_a_1"] = load_table(table_name="credit_bureau_a_1", files=config.DEPTH_1_TABLES_TRAIN, mode="train", prefix="credit_bureau_a")
            self.train_data["depth_1"]["train_credit_bureau_b_1"] = load_table(table_name="credit_bureau_b_1", files=config.DEPTH_1_TABLES_TRAIN, mode="train", prefix="credit_bureau_b")
            self.train_data["depth_1"]["train_deposit_1"] = load_table(table_name="deposit_1", files=config.DEPTH_1_TABLES_TRAIN, mode="train", prefix="deposit")
            self.train_data["depth_1"]["train_person_1"] = load_table(table_name="person_1", files=config.DEPTH_1_TABLES_TRAIN, mode="train", prefix="person")
            self.train_data["depth_1"]["train_debitcard_1"] = load_table(table_name="debitcard_1", files=config.DEPTH_1_TABLES_TRAIN, mode="train", prefix="debitcard")
            # # Depth 2
            self.train_data["depth_2"] = {}
            self.train_data["depth_2"]["train_applprev_2"] = load_table(table_name="applprev_2", files=config.DEPTH_2_TABLES_TRAIN, mode="train", prefix="applprev_2")
            self.train_data["depth_2"]["train_person_2"] = load_table(table_name="person_2", files=config.DEPTH_2_TABLES_TRAIN, mode="train", prefix="person")
            self.train_data["depth_2"]["train_credit_bureau_a_2"] = load_table(table_name="credit_bureau_a_2_0", files=config.DEPTH_2_TABLES_TRAIN, mode="train", prefix="credit_bureau_a")
            self.train_data["depth_2"]["train_credit_bureau_b_2"] = load_table(table_name="credit_bureau_b_2", files=config.DEPTH_2_TABLES_TRAIN, mode="train", prefix="credit_bureau_b")
        elif mode == "test":
            logger.info("Loading test data...")
        else:
            logger.error("Invalid mode. Choose from 'train' or 'test'.")
            return False
        
        logger.info("Data loaded successfully.")
        return True

    def drop_missing_value_data(self, mode="train"):
        """
        Preprocess data.
        """     
        if mode == "train":
            for key in self.train_data.keys():
                if key != "train_base":
                    for table in self.train_data[key].keys():
                        self.train_data[key][table] = self.train_data[key][table].select(self.filter_columns_by_missing_values(self.train_data[key][table], self.missing_threshold))

    def filter_columns_by_missing_values(self, data, threshold=0.3):
        """
        Filter columns by missing values using Polars.
        Parameters:
        - data: A Polars DataFrame
        - threshold: The maximum proportion of missing values allowed for keeping a column (default is 0.3, i.e., 30%)
        """
        # Calculate the proportion of missing values per column
        null_proportion = data.null_count() / data.shape[0]
        # Filter columns that have a missing value proportion less than the threshold (columns to keep)
        columns_to_keep = [col for col in data.columns if null_proportion[col][0] < threshold]

        return columns_to_keep

    # Calculate VIF for each feature
    def calculate_vif(self, df):
        vif_data = pd.DataFrame()
        vif_data["feature"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
        return vif_data

    def filter_VIF(self, data, threshold=5):
        """
        Filter columns by variance inflation factor (VIF) using Polars.
        Parameters:
        - data: A Polars DataFrame
        - threshold: The maximum VIF allowed for keeping a column (default is 5)
        """
        numeric_columns = [col for col in pl.DataFrame(data).columns if pl.DataFrame(data)[col].dtype in pl.NUMERIC_DTYPES]
        # Filter out features with high VIF values
        vif_threshold = threshold  # You can adjust this threshold
        vif_data = self.calculate_vif(data[numeric_columns])
        filtered_columns = vif_data[vif_data["VIF"] < vif_threshold]["feature"].tolist()
        return filtered_columns

    def filter_table(self,data):
        """
        Filter columns based on similar variables analysis.
        Keep only the recommended variables (Variable_To_Keep) from similar variables analysis.
        Parameters:
        --------
        data : DataFrame to filter
            
        Returns:
        --------
        data : Filtered DataFrame with only recommended variables
        """
        # Read the similar variables analysis
        similar_vars_df = pd.read_csv(config.SIMILAR_VARIABLES_ANALYSIS)
        # Get the list of variables to keep
        vars_to_keep = similar_vars_df['Variable_To_Keep'].tolist()
        
        # Get current columns
        current_columns = data.columns
        
        # Add required columns like case_id, target, etc. that should always be kept
        required_columns = config.KEY_COLUMNS
        keep_columns = [col for col in required_columns if col in current_columns]
        
        # Find which variables to keep in the current table
        keep_vars = [var for var in vars_to_keep if var in current_columns]
        
        # Add these to the keep_columns list
        return data.select(keep_columns.extend(keep_vars))

    def aggregate_data(self, mode="train"):
        """
        Aggregate data.
        """
        if mode == "train":
            for key in self.train_data.keys():
                if key != "train_base":
                    for table in self.train_data[key].keys():
                        df = self.train_data[key][table]
                        # Fill nulls with mean
                        df = Aggregator.fill_nulls(df, config.EXCLUDES)
                        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
                        # Convert categorical to numeric
                        df = Aggregator.convert_categorical_to_numeric(df, config.EXCLUDES)
                        self.train_data[key][table] = df
                        logger.info("Aggregated {} with shape: {}".format(table, df.shape))
        elif mode == "test":
            pass
        return True
    
    def join_data(self, mode="train"):
        """
        Join data.
        """
        # filter num_group1=0 and num_group2 =0
        for key in self.train_data.keys():
            if key == 'train_base':
                continue
            for table in self.train_data[key]:
                if "num_group1" in self.train_data[key][table].columns and "num_group2" in self.train_data[key][table].columns:
                    self.train_data[key][table] = self.train_data[key][table].filter(pl.col("num_group1") == 0).filter(pl.col("num_group2") == 0)
                if "num_group1" in self.train_data[key][table].columns:
                    self.train_data[key][table] = self.train_data[key][table].filter(pl.col("num_group1") == 0)
                if "num_group2" in self.train_data[key][table].columns:
                    self.train_data[key][table] = self.train_data[key][table].filter(pl.col("num_group2") == 0)

        if mode == "train":
            df_base = self.train_data["train_base"]
            # df_base = (
            #     df_base
            #     .with_columns(
            #             pl.col("date_decision").cast(pl.Date).alias("date_decision"),
            #             pl.col("date_decision").dt.month().alias("month_decision"),
            #             pl.col("date_decision").dt.weekday().alias("weekday_decision"),
            #     )
            # )            
            # Join all tables together.
            # data = df_base.join(
            #     self.train_data['depth_0']['train_static_cb_0'], how="left", on="case_id", suffix="_static_cb_0"
            # ).join(
            #     self.train_data['depth_1']['train_applprev_1'], how="left", on="case_id", suffix="_applprev_1"
            # ).join(
            #     self.train_data['depth_1']['train_other_1'], how="left", on="case_id", suffix="_other_1"
            # ).join(
            #     self.train_data['depth_1']['train_tax_registry_a_1'], how="left", on="case_id", suffix="_tax_registry_a_1"
            # ).join(
            #     self.train_data['depth_1']['train_tax_registry_b_1'], how="left", on="case_id", suffix="_tax_registry_b_1"
            # ).join(
            #     self.train_data['depth_1']['train_tax_registry_c_1'], how="left", on="case_id", suffix="_tax_registry_c_1"
            # ).join(
            #     self.train_data['depth_1']['train_deposit_1'], how="left", on="case_id", suffix="_deposit_1"
            # ).join(
            #     self.train_data['depth_1']['train_person_1'], how="left", on="case_id", suffix="_person_1"
            # ).join(
            #     self.train_data['depth_1']['train_debitcard_1'], how="left", on="case_id", suffix="_debitcard_1"
            # ).join(
            #     self.train_data['depth_2']['train_applprev_2'], how="left", on="case_id", suffix="_applprev_2"
            # ).join(
            #     self.train_data['depth_2']['train_person_2'], how="left", on="case_id", suffix="_person_2"
            # )

            data = df_base.join(
                self.train_data['depth_0']['train_static_0'], how="left", on="case_id", suffix="_static_0"
            ).join(
                self.train_data['depth_0']['train_static_cb_0'], how="left", on="case_id", suffix="_static_cb_0"
            ).join(
                self.train_data['depth_1']['train_applprev_1'], how="left", on="case_id", suffix="_applprev_1"
            ).join(
                self.train_data['depth_1']['train_other_1'], how="left", on="case_id", suffix="_other_1"
            ).join(
                self.train_data['depth_1']['train_tax_registry_a_1'], how="left", on="case_id", suffix="_tax_registry_a_1"
            ).join(
                self.train_data['depth_1']['train_tax_registry_b_1'], how="left", on="case_id", suffix="_tax_registry_b_1"
            ).join(
                self.train_data['depth_1']['train_tax_registry_c_1'], how="left", on="case_id", suffix="_tax_registry_c_1"
            ).join(
                self.train_data['depth_1']['train_credit_bureau_a_1'], how="left", on="case_id", suffix="_credit_bureau_a_1"
            ).join(
                self.train_data['depth_1']['train_credit_bureau_b_1'], how="left", on="case_id", suffix="_credit_bureau_b_1"
            ).join(
                self.train_data['depth_1']['train_deposit_1'], how="left", on="case_id", suffix="_deposit_1"
            ).join(
                self.train_data['depth_1']['train_person_1'], how="left", on="case_id", suffix="_person_1"
            ).join(
                self.train_data['depth_1']['train_debitcard_1'], how="left", on="case_id", suffix="_debitcard_1"
            ).join(
                self.train_data['depth_2']['train_applprev_2'], how="left", on="case_id", suffix="_applprev_2"
            ).join(
                self.train_data['depth_2']['train_person_2'], how="left", on="case_id", suffix="_person_2"
            ).join(
                self.train_data['depth_2']['train_credit_bureau_a_2'], how="left", on="case_id", suffix="_credit_bureau_a_2"
            ).join(
                self.train_data['depth_2']['train_credit_bureau_b_2'], how="left", on="case_id", suffix="_credit_bureau_b_2"
            )

            # df_base = data.pipe(handle_dates)
            del self.train_data
            # Filter missing values
            df_base = data.select(self.filter_columns_by_missing_values(data, threshold=config.MISSING_THRESHOLD))
            del data
            # Fill nulls
            df_base = Aggregator.fill_nulls(df_base, excludes=config.EXCLUDES)
            # Get limited sample of data
            df_base = df_base.sort(config.KEY_COLUMNS[2])
            df_base = df_base.sample(fraction=config.FRACTION_OF_DATA, with_replacement=False)
            logger.info("unique target: {}".format(df_base.to_pandas()['target'].value_counts()[[0, 1]]))
            logger.info("unique WEEK_NUM: {}".format(df_base.to_pandas()['WEEK_NUM'].nunique()))
            logger.info("sampled data with shape: {}".format(df_base.shape))
            gc.collect()
            df_base = df_base.sort(config.KEY_COLUMNS[2])
            # Filter VIF
            df_vif = df_base.drop(config.KEY_COLUMNS).to_pandas()
            df_base = df_base.to_pandas()
            df_base = df_base[config.KEY_COLUMNS + self.filter_VIF(df_vif, threshold=config.VIF_THRESHOLD)]
            logger.info("filtered VIF data with shape: {}".format(df_base.shape))
            save_dataframe_as_parquet(df_base, os.path.join(config.FEATURES_DIR,"train_base_VIF.parquet"))
            gc.collect()

            del df_vif
            return df_base
        
        elif mode == "test":
            pass



if __name__ == "__main__":
    # Preprocess data
    data = PreprocessData()
    # data.load_data()
    # data.preprocess_data()
    # print(data.train_data['Depth_0']['train_static_0'].head())



