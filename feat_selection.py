"""
Feature selection module.
"""
import pandas as pd
import numpy as np
import polars as pl
import logging
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import config
from utils import save_dataframe_as_parquet

class FeatureSelection:
    def __init__(self):
        pass
    
    def filter_multicollinearity(self, data):
        """
        Filter multicollinearity from the data.
        """
        # Create correlation matrix
        corr_matrix = data.corr(numeric_only=True).abs()
        high_corr_pairs = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > config.MULTICOLLINEARITY_THRESHOLD:
                    colname_i = corr_matrix.columns[i]
                    colname_j = corr_matrix.columns[j]
                    high_corr_pairs.add((colname_i, colname_j))

        # Drop one column from each pair
        columns_to_drop = set()
        for col1, col2 in high_corr_pairs:
            columns_to_drop.add(col2)  # You can choose to drop col1 or col2

        return data.drop(columns=columns_to_drop)  
    
    def correlation(self, data, target_column, num_features):
        X = data.drop(config.KEY_COLUMNS, axis=1)
        X = self.filter_multicollinearity(X)
        # Calculate the correlation matrix
        corr = X.corr(numeric_only=True)

        # Get the absolute correlations with 'target' and sort them
        # Only consider the absolute value of the correlation, and plot the heatmap
        # sorted_corr = corr[['target']].abs().sort_values(by='target', ascending=False)
        
        # Get the top features
        top_feat_corr = corr.head(num_features)
        return pl.DataFrame(data).select(config.KEY_COLUMNS + top_feat_corr.index.to_list())

    def lasso(self, data, target_column, num_features):
        data = pl.DataFrame(data)
        # Ensure the target column is present in the DataFrame
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        # Separate features and target
        X = data.drop(config.KEY_COLUMNS).to_numpy()
        y = data[target_column].to_numpy()

        # Standardize the features (important for Lasso)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit Lasso model
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_scaled, y)

        # Get feature importance
        feature_importance = abs(lasso.coef_)
        feature_names = data.drop(config.KEY_COLUMNS).columns

        # Select top features
        sorted_indices = feature_importance.argsort()[::-1][:num_features]
        selected_features = [feature_names[i] for i in sorted_indices]

        # Return the Polars DataFrame with only selected features
        return data.select(config.KEY_COLUMNS + selected_features)

    def random_forest(self, data, target_column, num_features):
        # Split data into features and target
        X = data.drop(config.KEY_COLUMNS,axis=1)
        y = data[target_column]
        
        # Train Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        feature_importances = rf.feature_importances_
        
        # Create a DataFrame with feature names and their importance
        feature_data = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importances
        })
        
        # Sort the features by importance and select the top `num_features`
        top_features = feature_data.sort_values(by='importance', ascending=False).head(num_features)
        
        # # Select only the top `num_features` from the original Polars DataFrame
        # selected_features = pl.DataFrame(data).select([col for col in top_features['feature']])
        
        return pl.DataFrame(data).select(config.KEY_COLUMNS + [col for col in top_features['feature']])
    
    def run_selection(self, data, methods=[], y='target'):
        """
        Run feature selection based on the method.
        """
        feats = {}
        for method in methods:
            logging.info(f"Running feature selection with {method}...")
            start_time = pd.Timestamp.now()

            if method == 'correlation':
                feats[method] = self.correlation(data, target_column=y, num_features=config.TOP_K_FEATS)
            elif method == 'lasso':   
                feats[method] = self.lasso(data, target_column=y, num_features=config.TOP_K_FEATS)
            elif method == 'random_forest':
                feats[method] = self.random_forest(data, target_column=y, num_features=config.TOP_K_FEATS)

            end_time = pd.Timestamp.now()
            duration = end_time - start_time
            logging.info(f"Feature selection with {method} took {duration}")
            save_dataframe_as_parquet(feats[method], os.path.join(config.FEATURES_DIR,f"selected-feats-{method}.parquet"))
        return feats
        # if method == 'correlation':
        #     return self.correlation(data)


