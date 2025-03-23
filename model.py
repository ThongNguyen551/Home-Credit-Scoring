# model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from matplotlib import pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import joblib
import config
import logging
from imblearn.over_sampling import SMOTE
from utils import save_dataframe_as_parquet

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

SAVE_FIG = True

class ModelTrainer:
    def __init__(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        oversample = SMOTE(sampling_strategy=0.5)
        X_train_balanced, y_train_balanced = oversample.fit_resample(X_train,y_train)
        save_dataframe_as_parquet(X_train_balanced, config.DATA_DIR / "X_train_balanced.parquet")
        save_dataframe_as_parquet(X_val, config.DATA_DIR / "X_val.parquet")
        save_dataframe_as_parquet(pd.DataFrame(y_train_balanced), config.DATA_DIR / "y_train_balanced.parquet")
        save_dataframe_as_parquet(pd.DataFrame(y_val), config.DATA_DIR / "y_val.parquet")

        if SAVE_FIG:
            # plot bar chart of X_train and X_train_balanced to compare
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].bar(['0', '1'], [len(y_train[y_train==0]), len(y_train[y_train==1])])
            ax[0].set_title('X_train')
            ax[1].bar(['0', '1'], [len(y_train_balanced[y_train_balanced==0]), len(y_train_balanced[y_train_balanced==1])])
            ax[1].set_title('X_train_balanced')
            plt.savefig(config.FIGURES_DIR / "train_vs_train_balanced.png")
            plt.close()

        self.df_train = X_train_balanced[config.KEY_COLUMNS]
        self.X_train = X_train_balanced.drop(config.KEY_COLUMNS, axis=1)
        self.y_train = y_train_balanced

        self.df_val = X_val[config.KEY_COLUMNS]
        self.X_val = X_val.drop(config.KEY_COLUMNS, axis=1)
        self.y_val = y_val

        self.results = {}
        self.best_models = {}
        self.models_params = {
            'Logistic Regression': {
                'model': LogisticRegression(verbose=0, random_state=123),
                'params': {
                    # 'C': [0.1, 1, 10], 
                    # 'solver': ['liblinear', 'lbfgs']
                    }
            },
            'LGBM': {
                'model': lgb.LGBMClassifier(device='gpu', verbose=-1, random_state=123),
                'params': {
                    "boosting_type": ["gbdt"],
                    "objective": ["binary"],
                    "metric": ["auc"],
                    "max_depth": [8,12],
                    "learning_rate": [0.05],
                    "n_estimators": [1000],
                    "colsample_bytree": [0.8], 
                    "colsample_bynode": [0.8]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(tree_method='hist', device='cuda', verbosity=0, random_state=123),
                'params': {
                    "booster": ["gbtree"],
                    "eval_metric": ["auc"],
                    'max_depth': [8, 12], 
                    'learning_rate': [0.05],
                    'n_estimators': [1000],
                    'colsample_bytree': [0.8],
                    'colsample_bynode': [0.8]
                    }
            },
            'CatBoost': {
                'model': cb.CatBoostClassifier(task_type="GPU", verbose=0, random_state=123),
                'params': {
                    "boosting_type": ["Plain"],
                    "eval_metric": ["AUC"],
                    "depth": [8],
                    "learning_rate": [0.05],
                    "iterations": [1000]
                }
            }
        }

    def save_model(self, model_name, model):
        model_path = config.MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"Model {model_name} saved to {model_path}")

    def mean_absolute_percentage_error(self, y_true, y_pred):
        epsilon = 1e-10  # Small value to avoid division by zero
        y_true = np.where(y_true == 0, epsilon, y_true)  # Replace zero values in y_true with epsilon
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def gini_stability(self, base, w_fallingrate=88.0, w_resstd=-0.5):
        # gini_in_time = base.loc[:, ["WEEK_NUM", "target", "score"]]\
        #     .sort_values("WEEK_NUM")\
        #     .groupby("WEEK_NUM")[["target", "score"]]\
        #     .apply(lambda x: 2 * roc_auc_score(x["target"], x["score"]) - 1).tolist()

        # Select relevant columns
        selected_columns = base.loc[:, ["WEEK_NUM", "target", "score"]]

        # Sort by 'WEEK_NUM'
        sorted_columns = selected_columns.sort_values("WEEK_NUM")

        # Group by 'WEEK_NUM'
        grouped = sorted_columns.groupby("WEEK_NUM")[["target", "score"]]

        # Calculate Gini coefficient for each group
        gini_in_time = []
        for name, group in grouped:
            if len(np.unique(group["target"])) > 1:
                roc_auc = roc_auc_score(group["target"], group["score"])
                gini = 2 * roc_auc - 1
            else:
                # logger.warning("Only one class is present in y_true. ROC AUC score is not defined in that case.")
                gini = 0  # or some other default value
            gini_in_time.append(gini)      

        # # Convert to list
        # gini_in_time = gini_in_time.tolist()

        x = np.arange(len(gini_in_time))
        y = gini_in_time
        a, b = np.polyfit(x, y, 1)
        y_hat = a * x + b
        residuals = y - y_hat
        res_std = np.std(residuals)
        avg_gini = np.mean(gini_in_time)
        return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std


    def train_models(self, model_names=None, feature_names=None):
        if model_names is None:
            model_names = self.models_params.keys()

        # X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        # oversample = SMOTE(sampling_strategy=0.5)
        # X_train_balanced, y_train_balanced = oversample.fit_resample(self.X_train,self.y_train)

        for model_name in model_names:
            if model_name not in self.models_params:
                raise ValueError(f"Model {model_name} is not recognized.")

            start_time = pd.Timestamp.now()
            logger.info(f"Training {model_name}...")

            mp = self.models_params[model_name]
            grid_search = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='accuracy')
            grid_search.fit(self.X_train, self.y_train)
            model = grid_search.best_estimator_

            end_time = pd.Timestamp.now()
            duration = end_time - start_time
            logger.info(f"Training {model_name} complete. Duration: {duration}")

            y_pred = model.predict(self.X_val)
            y_pred_proba = model.predict_proba(self.X_val)[:, 1]
            y_pred_proba_train = model.predict_proba(self.X_train)[:, 1]

            self.df_train['score']=y_pred_proba_train
            self.df_val['score']=y_pred_proba
            metrics = {
                'accuracy': accuracy_score(self.y_val, y_pred),  
                'precision': precision_score(self.y_val, y_pred),
                'recall': recall_score(self.y_val, y_pred),
                'f1_score': f1_score(self.y_val, y_pred),
                'auc': roc_auc_score(self.y_val, y_pred_proba) if len(np.unique(self.y_val)) > 1 else None,
                'train_gini': 2 * roc_auc_score(self.y_train, y_pred_proba_train) - 1,
                'val_gini': 2 * roc_auc_score(self.y_val, y_pred_proba) - 1, 
                'train_gini_stability': self.gini_stability(self.df_train),
                'val_gini_stability': self.gini_stability(self.df_val)
            }

            self.best_models[model_name] = model
            self.results[model_name] = metrics

            # get current date and time to name model
            current_date = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            # save the model
            self.save_model(f"{model_name}_{feature_names}_{current_date}", model)
            # Print the predictions
            # print(f"Predictions for {model_name}: {y_pred_val}")
            # print(f"Predictions proba for {model_name}: {y_pred_proba_val}")

        return self.results
    
    def evaluate(self, model_name, X_test, y_test):
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} has not been trained.")

        model = self.best_models[model_name]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }


    def predict(self, model_name, X):
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} has not been trained.")

        model = self.best_models[model_name]
        return model.predict(X)

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)

    trainer = ModelTrainer(X, y)
    results = trainer.train_models(['SVM', 'XGBoost'])
    for model, res in results.items():
        print(f"{model}: Accuracy = {res['accuracy']:.4f}, Precision = {res['precision']:.4f}, Recall = {res['recall']:.4f}, F1-score = {res['f1_score']:.4f}, AUC = {res['auc']}")

