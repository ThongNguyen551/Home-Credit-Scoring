"""
This file contains utility functions that are used in the main script.
"""
import numpy as np
import pandas as pd
import polars as pl
import logging
import os
from pathlib import Path

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)=="category":
            continue
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            continue
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

import polars as pl
import numpy as np

def reduce_mem_usage_pl(df: pl.DataFrame) -> pl.DataFrame:
    """
    Iterate through all columns in a Polars DataFrame and modify the data type
    to reduce memory usage, while handling null values.
    """
    # Initial memory usage estimate
    start_mem = df.estimated_size() / 1024**2  # In MB
    print(f"Memory usage of dataframe: {start_mem:.2f} MB")
    
    # Process each column
    new_cols = []
    for col in df.columns:
        col_type = df[col].dtype
        
        # Skip non-numeric columns
        if col_type.is_numeric():
            # Handle null-only columns
            if df[col].null_count() == df.height:
                new_cols.append(df[col].cast(pl.Float32))  # Or keep as is
                continue
            
            # Calculate min and max, handling nulls
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min is None or col_max is None:
                new_cols.append(df[col])
                continue

            # Handle integer columns
            if col_type.is_integer():
                if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                    new_cols.append(df[col].cast(pl.Int8))
                elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    new_cols.append(df[col].cast(pl.Int16))
                elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    new_cols.append(df[col].cast(pl.Int32))
                else:
                    new_cols.append(df[col].cast(pl.Int64))
            
            # Handle float columns
            elif col_type.is_float():
                if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                    new_cols.append(df[col].cast(pl.Float32))
                else:
                    new_cols.append(df[col].cast(pl.Float64))

        # Keep non-numeric columns unchanged
        else:
            new_cols.append(df[col])
    
    # Reconstruct the DataFrame
    df_optimized = pl.DataFrame(new_cols)
    
    # Final memory usage
    end_mem = df_optimized.estimated_size() / 1024**2
    print(f"Memory usage after optimization: {end_mem:.2f} MB")
    print(f"Reduced by: {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df_optimized

def reduce_mem_usage_pl_custom(df: pl.DataFrame) -> pl.DataFrame:
    """
    Iterate through all the columns of a Polars DataFrame and modify the data type
    to reduce memory usage.
    """
    # Initial memory usage
    start_mem = df.estimated_size() / 1024**2  # In MB
    # print(f"Memory usage of dataframe: {start_mem:.2f} MB")
    
    # Define the min and max values for each integer type
    int8_min, int8_max = -128, 127
    int16_min, int16_max = -32768, 32767
    int32_min, int32_max = -2147483648, 2147483647
    int64_min, int64_max = -9223372036854775808, 9223372036854775807

    # Process each column
    new_cols = []
    for col in df.columns:
        col_type = df[col].dtype

        # Skip non-numeric columns (like strings or categories)
        if col_type.is_numeric():
            col_min = df[col].min()
            col_max = df[col].max()
            
            # Handle None values
            if col_min is None or col_max is None:
                new_cols.append(df[col])
                continue
            
            # For integer columns
            if col_type.is_integer():
                if col_min >= int8_min and col_max <= int8_max:
                    new_cols.append(df[col].cast(pl.Int8))
                elif col_min >= int16_min and col_max <= int16_max:
                    new_cols.append(df[col].cast(pl.Int16))
                elif col_min >= int32_min and col_max <= int32_max:
                    new_cols.append(df[col].cast(pl.Int32))
                else:
                    new_cols.append(df[col].cast(pl.Int64))

            # For float columns
            elif col_type.is_float():
                if col_min >= -3.4e38 and col_max <= 3.4e38:
                    new_cols.append(df[col].cast(pl.Float32))
                else:
                    new_cols.append(df[col].cast(pl.Float64))

        # Keep non-numeric columns unchanged
        else:
            new_cols.append(df[col])
    
    # Create a new DataFrame
    df_optimized = pl.DataFrame(new_cols)
    
    # Final memory usage
    end_mem = df_optimized.estimated_size() / 1024**2
    # print(f"Memory usage after optimization: {end_mem:.2f} MB")
    # print(f"Reduced by: {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df_optimized

def set_table_dtypes(df): #Standardize the dtype.
    for col in df.columns:
        if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
            df = df.with_columns(pl.col(col).cast(pl.Int64))
        elif col in ["date_decision"]:
            df = df.with_columns(pl.col(col).cast(pl.Date))
        elif col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64))
        elif col[-1] in ("M",):
            df = df.with_columns(pl.col(col).cast(pl.String))
        elif col[-1] in ("D",):
            df = df.with_columns(pl.col(col).cast(pl.Date))            

    return df

def table_to_dict(df: pl.DataFrame) -> dict:
    """
    Convert a Polars DataFrame with 2 columns into a dictionary.
    The first column will be the keys and the second column will be the values.
    """
    if df.shape[1] != 2:
        raise ValueError("DataFrame must have exactly 2 columns")

    # Convert the DataFrame to a dictionary
    result_dict = dict(zip(df[:, 0], df[:, 1]))
    
    return result_dict

# def handle_dates(df):
#     for col in df.columns:
#         if col[-1] in ("D",):
#             df = df.with_columns(pl.col(col) - pl.col("date_decision"))  #!!?
#             df = df.with_columns(pl.col(col).dt.total_days()) # t - t-1
#     df = df.drop("date_decision", "MONTH")
#     return df

def handle_dates(df):
    for col in df.columns:
        if col.endswith("D"):
            # Ensure the date_decision column is in the correct format
            df = df.with_columns(pl.col("date_decision").cast(pl.Date))
            # Subtract date_decision from the column
            df = df.with_columns((pl.col(col) - pl.col("date_decision")).alias(col))
            # Convert the resulting timedelta to total days
            df = df.with_columns(pl.col(col).dt.days().alias(col))
    # Drop the date_decision and MONTH columns
    df = df.drop(["date_decision", "MONTH"])
    return df

def filter_cols(df):
    for col in df.columns:
        if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
            freq = df[col].n_unique()
            if (freq == 1) | (freq > 200):
                df = df.drop(col)

def save_dataframe_as_parquet(df, filename="data.parquet", engine="pyarrow", compression="snappy"):
    """
    Checks if the DataFrame is valid and saves it as a Parquet file.
    
    Supports both Pandas and Polars DataFrames.
    
    Parameters:
        df (pd.DataFrame or pl.DataFrame): The DataFrame to save.
        filename (str): The name of the output file.
        engine (str): Parquet engine to use for Pandas ("pyarrow" or "fastparquet").
        compression (str): Compression method ("snappy", "gzip", "brotli", etc.).
    
    Returns:
        str: Success message or error.
    """
    
    if df is None or len(df) == 0:
        return "Error: DataFrame is empty or None."
    
    if isinstance(df, pl.DataFrame):
        try:
            df.write_parquet(filename, compression=compression)
            return f"Polars DataFrame saved as {filename} with {compression} compression."
        except Exception as e:
            return f"Error saving Polars DataFrame: {e}"
    
    elif isinstance(df, pd.DataFrame):
        try:
            df.to_parquet(filename, engine=engine, compression=compression)
            return f"Pandas DataFrame saved as {filename} using {engine} with {compression} compression."
        except Exception as e:
            return f"Error saving Pandas DataFrame: {e}"
    
    else:
        return "Error: Unsupported DataFrame type. Use Pandas or Polars."