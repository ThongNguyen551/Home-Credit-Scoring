import joblib
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix, 
    precision_recall_curve, average_precision_score
)
import logging
import config

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, models_dir=None):
        """
        Initialize the ModelEvaluator class.
        
        Parameters:
        -----------
        models_dir : str or Path, optional
            Directory containing saved model files (.pkl)
        """
        self.models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
        self.models = {}
        self.results = {}
        
        # Create figures directory if it doesn't exist
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        
    def load_model(self, model_path):
        """
        Load a single model from the specified path.
        
        Parameters:
        -----------
        model_path : str or Path
            Path to the model file (.pkl)
            
        Returns:
        --------
        The loaded model
        """
        try:
            model_path = Path(model_path)
            model = joblib.load(model_path)
            model_name = model_path.stem
            self.models[model_name] = model
            logger.info(f"Model {model_name} loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None
    
    def load_models(self, pattern=None, feature_selection_technique=None):
        """
        Load all models from the models directory matching the given pattern.
        
        Parameters:
        -----------
        pattern : str, optional
            Pattern to match model filenames (e.g., 'LGBM_*')
            
        Returns:
        --------
        dict: Dictionary of loaded models
        """
        if not self.models_dir.exists():
            logger.error(f"Models directory {self.models_dir} does not exist")
            return {}

        # Define search pattern based on inputs
        if feature_selection_technique:
            search_pattern = f"*{feature_selection_technique}*.pkl"
            logger.info(f"Loading models with feature selection technique: {feature_selection_technique}")
        elif pattern:
            search_pattern = pattern
        else:
            search_pattern = "*.pkl"

        # search_pattern = pattern if pattern else "*.pkl"
        model_files = list(self.models_dir.glob(search_pattern))
        
        if not model_files:
            logger.warning(f"No model files found matching pattern '{search_pattern}' in {self.models_dir}")
            return {}
        
        for model_file in model_files:
            self.load_model(model_file)
            
        return self.models
    
    def gini_coefficient(self, y_true, y_pred_proba):
        """Calculate Gini coefficient (normalized AUC*2-1)"""
        return 2 * roc_auc_score(y_true, y_pred_proba) - 1
    
    def gini_stability(self, base, w_fallingrate=88.0, w_resstd=-0.5):
        """Calculate Gini stability over time (similar to the ModelTrainer implementation)"""
        selected_columns = base.loc[:, ["WEEK_NUM", "target", "score"]]
        sorted_columns = selected_columns.sort_values("WEEK_NUM")
        grouped = sorted_columns.groupby("WEEK_NUM")[["target", "score"]]
        
        gini_in_time = []
        for name, group in grouped:
            if len(np.unique(group["target"])) > 1:
                roc_auc = roc_auc_score(group["target"], group["score"])
                gini = 2 * roc_auc - 1
            else:
                gini = 0
            gini_in_time.append(gini)      
        
        x = np.arange(len(gini_in_time))
        y = gini_in_time
        a, b = np.polyfit(x, y, 1)
        y_hat = a * x + b
        residuals = y - y_hat
        res_std = np.std(residuals)
        avg_gini = np.mean(gini_in_time)
        return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std
    
    def evaluate_model(self, model_name, X, y, key_columns=None):
        """
        Evaluate a loaded model on the provided dataset.
        
        Parameters:
        -----------
        model_name : str
            Name/key of the model in the models dictionary
        X : DataFrame
            Features for evaluation
        y : Series or array
            Target values for evaluation
        key_columns : list, optional
            List of key columns to keep (like 'case_id', 'WEEK_NUM')
            
        Returns:
        --------
        dict: Dictionary of evaluation metrics
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found in loaded models")
            return None
        
        model = self.models[model_name]
        
        # Separate key columns if specified
        df_eval = None
        X_eval = X
        if key_columns is not None and set(key_columns).issubset(X.columns):
            df_eval = X[key_columns].copy()
            X_eval = X.drop(key_columns, axis=1)
        

        # Generate predictions
        y_pred = model.predict(X_eval)
        y_pred_proba = model.predict_proba(X_eval)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_pred_proba),
            'gini': self.gini_coefficient(y, y_pred_proba),
            'avg_precision': average_precision_score(y, y_pred_proba)
        }
        
        # # Calculate gini stability if we have WEEK_NUM
        # if df_eval is not None and 'WEEK_NUM' in df_eval.columns:
        #     df_eval['score'] = y_pred_proba
        #     df_eval['target'] = y
        #     metrics['gini_stability'] = self.gini_stability(df_eval)
        
        self.results[model_name] = metrics
        return metrics
    
    def evaluate_all_models(self, X, y, key_columns=None):
        """
        Evaluate all loaded models on the provided dataset.
        
        Parameters:
        -----------
        X : DataFrame
            Features for evaluation
        y : Series or array
            Target values for evaluation
        key_columns : list, optional
            List of key columns to keep (like 'case_id', 'WEEK_NUM')
            
        Returns:
        --------
        DataFrame: Results for all models
        """
        for model_name in self.models:
            self.evaluate_model(model_name, X, y, key_columns)
            
        return pd.DataFrame(self.results).T
    
    def plot_roc_curves(self, X, y):
        """
        Plot ROC curves for all models.
        
        Parameters:
        -----------
        X : DataFrame
            Features for evaluation
        y : Series or array
            Target values for evaluation
        """
        plt.figure(figsize=(12, 8))
        
        for model_name, model in self.models.items():
            X_eval = X
            if set(config.KEY_COLUMNS).issubset(X.columns):
                X_eval = X.drop(config.KEY_COLUMNS, axis=1)
                
            y_pred_proba = model.predict_proba(X_eval)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            auc_score = roc_auc_score(y, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(config.FIGURES_DIR / "roc_curves_comparison.png")
        plt.show()
    
    def plot_precision_recall_curves(self, X, y):
        """
        Plot Precision-Recall curves for all models.
        
        Parameters:
        -----------
        X : DataFrame
            Features for evaluation
        y : Series or array
            Target values for evaluation
        """
        plt.figure(figsize=(12, 8))
        
        for model_name, model in self.models.items():
            X_eval = X
            if set(config.KEY_COLUMNS).issubset(X.columns):
                X_eval = X.drop(config.KEY_COLUMNS, axis=1)
                
            y_pred_proba = model.predict_proba(X_eval)[:, 1]
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)
            avg_precision = average_precision_score(y, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.4f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for All Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(config.FIGURES_DIR / "precision_recall_curves_comparison.png")
        plt.show()
    
    def plot_confusion_matrix(self, model_name, X, y, threshold=0.5):
        """
        Plot confusion matrix for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name/key of the model in the models dictionary
        X : DataFrame
            Features for evaluation
        y : Series or array
            Target values for evaluation
        threshold : float, optional
            Threshold for binary classification
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found in loaded models")
            return
        
        model = self.models[model_name]
        
        X_eval = X
        if set(config.KEY_COLUMNS).issubset(X.columns):
            X_eval = X.drop(config.KEY_COLUMNS, axis=1)
            
        y_pred_proba = model.predict_proba(X_eval)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Save the plot
        plt.savefig(config.FIGURES_DIR / f"confusion_matrix_{model_name}.png")
        plt.show()
    
    def feature_importance(self, model_name, feature_names=None):
        """
        Plot feature importance for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name/key of the model in the models dictionary
        feature_names : list, optional
            List of feature names
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found in loaded models")
            return
            
        model = self.models[model_name]
        
        # Get feature importance based on model type
        importance = None
        model_type = type(model).__name__
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                logger.warning(f"Model {model_name} doesn't have feature importance attributes")
                return
                
            if feature_names is None or len(feature_names) != len(importance):
                feature_names = [f"Feature {i}" for i in range(len(importance))]
                
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 10))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
            plt.title(f'Top 20 Feature Importance - {model_name}')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(config.FIGURES_DIR / f"feature_importance_{model_name}.png")
            plt.show()
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {model_name}: {e}")
            return None

# Example usage
def run_model_evaluation(test_data_path, feature_names=None):
    """
    Run evaluation on test data for all saved models.
    
    Parameters:
    -----------
    test_data_path : str or Path
        Path to test data CSV file
    feature_names : list, optional
        List of feature names
    """
    # Load test data
    test_data = pd.read_parquet(test_data_path)
    
    # Ensure WEEK_NUM column is present (for stability calculation)
    if 'date_decision' in test_data.columns and 'WEEK_NUM' not in test_data.columns:
        test_data['date_decision'] = pd.to_datetime(test_data['date_decision'])
        test_data['WEEK_NUM'] = test_data['date_decision'].dt.isocalendar().week
    
    # Separate features and target
    y = test_data[config.KEY_COLUMNS[2]]
    X = test_data.drop(config.KEY_COLUMNS, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    oversample = SMOTE(sampling_strategy=0.5)
    X_train_balanced, y_train_balanced = oversample.fit_resample(X_train, y_train)

    # Initialize evaluator and load all models
    evaluator = ModelEvaluator(models_dir=config.MODELS_DIR)
    evaluator.load_models(feature_selection_technique='random_forest')
    
    # Evaluate all models
    datasets = [
        ("Training", X_train_balanced, y_train_balanced),
        ("Validation", X_val, y_val)
    ]
    
    results = {}
    for dataset_name, X_test, y_test in datasets:
        # Create a copy of the evaluator for each dataset to avoid mixing results
        dataset_evaluator = ModelEvaluator(models_dir=config.MODELS_DIR)
        dataset_evaluator.models = evaluator.models.copy()
        
        eval_results = dataset_evaluator.evaluate_all_models(X_test, y_test, key_columns=config.KEY_COLUMNS)
        print(f"\n{dataset_name} Set - Model Evaluation Results:")
        print(eval_results)
        results[dataset_name] = eval_results
        
        # Update plot titles and filenames to include dataset name
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        for model_name, model in dataset_evaluator.models.items():
            X_eval = X_test
            if set(config.KEY_COLUMNS).issubset(X_test.columns):
                X_eval = X_test.drop(config.KEY_COLUMNS, axis=1)
                
            y_pred_proba = model.predict_proba(X_eval)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {dataset_name} Set')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(config.FIGURES_DIR / f"roc_curves_{dataset_name.lower()}.png")
        plt.close()
        
        # Plot Precision-Recall curves
        plt.figure(figsize=(12, 8))
        for model_name, model in dataset_evaluator.models.items():
            X_eval = X_test
            if set(config.KEY_COLUMNS).issubset(X_test.columns):
                X_eval = X_test.drop(config.KEY_COLUMNS, axis=1)
                
            y_pred_proba = model.predict_proba(X_eval)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.4f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - {dataset_name} Set')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(config.FIGURES_DIR / f"precision_recall_curves_{dataset_name.lower()}.png")
        plt.close()
        
        # Plot confusion matrix for each model
        for model_name in dataset_evaluator.models:
            if model_name not in dataset_evaluator.models:
                continue
            
            model = dataset_evaluator.models[model_name]
            
            X_eval = X_test
            if set(config.KEY_COLUMNS).issubset(X_test.columns):
                X_eval = X_test.drop(config.KEY_COLUMNS, axis=1)
                
            y_pred_proba = model.predict_proba(X_eval)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Confusion Matrix - {model_name} ({dataset_name} Set)')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(config.FIGURES_DIR / f"confusion_matrix_{model_name}_{dataset_name.lower()}.png")
            plt.close()
            
            # Plot feature importance if feature_names are provided
            if feature_names is not None:
                try:
                    model = dataset_evaluator.models[model_name]
                    
                    # Get feature importance based on model type
                    importance = None
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        importance = np.abs(model.coef_[0])
                    else:
                        continue
                        
                    if feature_names is None or len(feature_names) != len(importance):
                        feature_names = [f"Feature {i}" for i in range(len(importance))]
                        
                    # Create DataFrame for plotting
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    # Plot top 20 features
                    plt.figure(figsize=(12, 10))
                    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
                    plt.title(f'Top 20 Feature Importance - {model_name} ({dataset_name} Set)')
                    plt.tight_layout()
                    plt.savefig(config.FIGURES_DIR / f"feature_importance_{model_name}_{dataset_name.lower()}.png")
                    plt.close()
                except Exception as e:
                    logger.error(f"Error plotting feature importance for {model_name}: {e}")
    
    return results

if __name__ == "__main__":
    fpath = "C:\\Users\\thong\\Desktop\\Home_Credit\\data\\features\\selected-feats-random_forest.parquet"
    test_data_path = glob.glob(fpath)[0]  # Take the first match since glob returns a list
    
    # Example feature names (you should replace with your actual feature names)
    feature_names = None  # If you have the feature names, place them here
    
    # Run the evaluation
    results = run_model_evaluation(test_data_path, feature_names)