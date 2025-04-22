"""
Machine Learning Utilities
=========================

Comprehensive toolkit for machine learning tasks including:
    - Data preprocessing
    - Model evaluation
    - Visualization
    - Feature engineering
    - Cross-validation

This module simplifies common ML workflows and provides
consistent interfaces for various ML tasks.
"""


import pandas as pd
import numpy as np
import joblib  # Preferred over pickle for numpy arrays often found in sklearn models
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.base import BaseEstimator # For custom transformers if needed

# --- Configuration ---
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates machine learning models with various metrics and visualizations.
    
    Methods:
        calculate_metrics: Computes classification/regression metrics
        plot_confusion_matrix: Visualizes confusion matrix
        cross_validate: Performs k-fold cross-validation
        
    Example:
        >>> evaluator = ModelEvaluator()
        >>> metrics = evaluator.calculate_metrics(y_true, y_pred)
        >>> print(metrics)
        {'accuracy': 0.95, 'precision': 0.94, ...}
    """

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculates comprehensive classification metrics.
        
        Parameters:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            dict: Dictionary containing various metrics
                - accuracy
                - precision
                - recall
                - f1_score
                
        Example:
            >>> metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
            >>> print(f"Accuracy: {metrics['accuracy']:.2f}")
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }



class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        
    def scale_features(self, X: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Scale features using specified method."""
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        return self.scaler.fit_transform(X)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, 
                   random_state: int = 42) -> tuple:
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, 
                              random_state=random_state)

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

class Visualizer:
    @staticmethod
    def plot_feature_importance(feature_importance: np.ndarray, 
                              feature_names: List[str]) -> None:
        """Plot feature importance."""
        plt.figure(figsize=(10, 6))
        indices = np.argsort(feature_importance)[::-1]
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), 
                feature_importance[indices])
        plt.xticks(range(len(indices)), 
                   [feature_names[i] for i in indices], 
                   rotation=45)
        plt.tight_layout()
        plt.show()
        


# === 1. Data Loading ===

def load_csv_data(filepath: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.
        **kwargs: Additional keyword arguments passed to pandas.read_csv.

    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame, or None if loading fails.
    """
    try:
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Error: No data found in {filepath}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {filepath}: {e}")
        return None

# === 2. Data Preprocessing ===

def create_preprocessing_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
    num_imputation_strategy: str = 'median',
    cat_imputation_strategy: str = 'most_frequent',
    scaler: Optional[BaseEstimator] = StandardScaler(), # e.g., StandardScaler(), MinMaxScaler(), None
    handle_unknown_categories: str = 'ignore' # For OneHotEncoder
) -> Pipeline:
    """
    Creates a scikit-learn preprocessing pipeline using ColumnTransformer.

    Handles imputation, scaling (for numerical), and one-hot encoding (for categorical).

    Args:
        numerical_features (List[str]): List of numerical column names.
        categorical_features (List[str]): List of categorical column names.
        num_imputation_strategy (str): Strategy for numerical imputation ('mean', 'median', 'most_frequent', 'constant').
        cat_imputation_strategy (str): Strategy for categorical imputation ('most_frequent', 'constant').
        scaler (Optional[BaseEstimator]): Scaler instance for numerical features (e.g., StandardScaler(), MinMaxScaler()). If None, no scaling is applied.
        handle_unknown_categories (str): How OneHotEncoder handles unknown categories ('error' or 'ignore').

    Returns:
        Pipeline: A scikit-learn Pipeline object ready to be fit/transformed.
    """
    numerical_steps = [
        ('imputer', SimpleImputer(strategy=num_imputation_strategy))
    ]
    if scaler:
        numerical_steps.append(('scaler', scaler))
    numerical_transformer = Pipeline(steps=numerical_steps)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cat_imputation_strategy, fill_value='missing')), # fill_value needed for constant
        ('onehot', OneHotEncoder(handle_unknown=handle_unknown_categories, sparse_output=False)) # sparse=False often easier for downstream tasks
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns not specified
    )

    # Wrap ColumnTransformer in a Pipeline for easier use
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    logger.info("Created preprocessing pipeline.")
    return pipeline

# === 3. Data Splitting ===

def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target variable column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.
        stratify (bool): If True, data is split in a stratified fashion using the target column. Recommended for classification.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    stratify_param = y if stratify and y.nunique() > 1 else None # Stratify only if possible and requested

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    logger.info(f"Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# === 4. Model Training ===

def train_model(
    model: BaseEstimator,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    preprocessing_pipeline: Optional[Pipeline] = None
) -> BaseEstimator:
    """
    Trains a machine learning model, optionally applying preprocessing first.

    Args:
        model (BaseEstimator): The scikit-learn compatible model instance to train.
        X_train (Union[pd.DataFrame, np.ndarray]): Training features.
        y_train (Union[pd.Series, np.ndarray]): Training target variable.
        preprocessing_pipeline (Optional[Pipeline]): A fitted preprocessing pipeline to apply before training.
                                                    If None, assumes X_train is already preprocessed.

    Returns:
        BaseEstimator: The trained model (or a pipeline including the model if preprocessing was provided).
    """
    if preprocessing_pipeline:
        # Create a full pipeline including the model
        full_pipeline = Pipeline(steps=[
            ('preprocessing', preprocessing_pipeline),
            ('model', model)
        ])
        logger.info(f"Training model {type(model).__name__} within a full pipeline...")
        full_pipeline.fit(X_train, y_train)
        logger.info("Model training complete within pipeline.")
        return full_pipeline # Return the entire pipeline
    else:
        logger.info(f"Training model {type(model).__name__} directly...")
        model.fit(X_train, y_train)
        logger.info("Model training complete.")
        return model # Return just the trained model

# === 5. Model Evaluation ===

def evaluate_classification(
    model: BaseEstimator,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    decision_threshold: Optional[float] = None # For binary classification ROC AUC
) -> Dict[str, Any]:
    """
    Evaluates a classification model.

    Args:
        model (BaseEstimator): The trained model or pipeline.
        X_test (Union[pd.DataFrame, np.ndarray]): Test features.
        y_test (Union[pd.Series, np.ndarray]): True test labels.
        decision_threshold (Optional[float]): Threshold for converting probabilities to class labels (default 0.5 if needed).

    Returns:
        Dict[str, Any]: Dictionary containing various classification metrics.
    """
    logger.info("Evaluating classification model...")
    y_pred = model.predict(X_test)
    metrics = {}

    try:
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        # Calculate other metrics based on problem type (binary/multiclass)
        avg_method = 'binary' if y_test.nunique() <= 2 else 'weighted' # Adjust averaging for multiclass
        metrics['precision'] = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        metrics['f1_score'] = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist() # Convert to list for easier serialization

        # ROC AUC requires probability scores (binary or multiclass OvR)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2: # Binary classification
                 metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            else: # Multiclass classification
                 metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average=avg_method)

    except Exception as e:
        logger.error(f"Error during classification evaluation: {e}")
        metrics['error'] = str(e)

    logger.info(f"Classification Metrics: { {k: v for k, v in metrics.items() if k != 'confusion_matrix'} }") # Log without large matrix
    return metrics

def evaluate_regression(
    model: BaseEstimator,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Evaluates a regression model.

    Args:
        model (BaseEstimator): The trained model or pipeline.
        X_test (Union[pd.DataFrame, np.ndarray]): Test features.
        y_test (Union[pd.Series, np.ndarray]): True test target values.

    Returns:
        Dict[str, float]: Dictionary containing various regression metrics.
    """
    logger.info("Evaluating regression model...")
    y_pred = model.predict(X_test)
    metrics = {}
    try:
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2_score'] = r2_score(y_test, y_pred)
    except Exception as e:
        logger.error(f"Error during regression evaluation: {e}")
        metrics['error'] = str(e)

    logger.info(f"Regression Metrics: {metrics}")
    return metrics

# === 6. Hyperparameter Tuning ===

def tune_hyperparameters(
    model: BaseEstimator,
    param_grid: Dict[str, List[Any]],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    scoring: Optional[str] = None, # e.g., 'accuracy', 'f1', 'neg_mean_squared_error', 'r2'
    cv: int = 5,
    strategy: str = 'grid', # 'grid' or 'random'
    n_iter: int = 10, # Used only for RandomizedSearchCV
    n_jobs: int = -1, # Use all available CPU cores
    preprocessing_pipeline: Optional[Pipeline] = None # Apply preprocessing within CV if provided
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Performs hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

    Args:
        model (BaseEstimator): The model instance to tune.
        param_grid (Dict[str, List[Any]]): Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.
        X_train (Union[pd.DataFrame, np.ndarray]): Training features.
        y_train (Union[pd.Series, np.ndarray]): Training target variable.
        scoring (Optional[str]): Scoring metric to evaluate parameter combinations. Uses model's default if None.
        cv (int): Number of cross-validation folds.
        strategy (str): 'grid' for GridSearchCV, 'random' for RandomizedSearchCV.
        n_iter (int): Number of parameter settings sampled for RandomizedSearchCV.
        n_jobs (int): Number of jobs to run in parallel (-1 means using all processors).
        preprocessing_pipeline (Optional[Pipeline]): If provided, tuning is done on a pipeline including preprocessing. Parameter keys in param_grid should reflect this (e.g., 'model__parameter_name').

    Returns:
        Tuple[BaseEstimator, Dict[str, Any]]: The best estimator found and its parameters.
    """
    if preprocessing_pipeline:
        # Create a full pipeline for tuning
        pipeline_to_tune = Pipeline(steps=[
            ('preprocessing', preprocessing_pipeline),
            ('model', model)
        ])
        # Adjust param_grid keys to target model parameters within the pipeline
        pipeline_param_grid = {f'model__{k}': v for k, v in param_grid.items()}
        target_estimator = pipeline_to_tune
        param_grid_to_use = pipeline_param_grid
        logger.info(f"Tuning hyperparameters for {type(model).__name__} within a pipeline...")
    else:
        target_estimator = model
        param_grid_to_use = param_grid
        logger.info(f"Tuning hyperparameters for {type(model).__name__} directly...")


    if strategy == 'grid':
        search = GridSearchCV(
            estimator=target_estimator,
            param_grid=param_grid_to_use,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1 # Add some verbosity
        )
    elif strategy == 'random':
        search = RandomizedSearchCV(
            estimator=target_estimator,
            param_distributions=param_grid_to_use,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=DEFAULT_RANDOM_STATE,
            verbose=1
        )
    else:
        raise ValueError("Strategy must be 'grid' or 'random'")

    try:
        search.fit(X_train, y_train)
        logger.info(f"Hyperparameter tuning complete. Best score ({scoring or 'default'}): {search.best_score_:.4f}")
        logger.info(f"Best parameters found: {search.best_params_}")
        return search.best_estimator_, search.best_params_
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        # Return the original model and empty params on error
        return target_estimator, {}


# === 7. Model Persistence ===

def save_model(model: BaseEstimator, filepath: str) -> bool:
    """
    Saves a trained model (or pipeline) to a file using joblib.

    Args:
        model (BaseEstimator): The trained model or pipeline object.
        filepath (str): The path where the model should be saved (e.g., 'model.joblib').

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        joblib.dump(model, filepath)
        logger.info(f"Model successfully saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {e}")
        return False

def load_model(filepath: str) -> Optional[BaseEstimator]:
    """
    Loads a model (or pipeline) from a file using joblib.

    Args:
        filepath (str): The path to the saved model file.

    Returns:
        Optional[BaseEstimator]: The loaded model object, or None if loading fails.
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model successfully loaded from {filepath}")
        return model
    except FileNotFoundError:
        logger.error(f"Error: Model file not found at {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {e}")
        return None


# === Example Usage ===
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.datasets import make_classification, make_regression

    logger.info("--- Running Example Usage ---")

    # --- Classification Example ---
    logger.info("\n--- Classification Example ---")
    # 1. Create synthetic data
    X_cls, y_cls = make_classification(n_samples=200, n_features=10, n_informative=5, n_redundant=2,
                                       n_classes=2, random_state=DEFAULT_RANDOM_STATE)
    # Convert to DataFrame (more realistic) - 3 categorical, 7 numerical
    feature_names = [f'num_{i}' for i in range(7)] + [f'cat_{i}' for i in range(3)]
    X_cls_df = pd.DataFrame(X_cls, columns=feature_names)
    # Make some features categorical (example: convert float ranges to categories)
    for i in range(3):
        cat_col = f'cat_{i}'
        X_cls_df[cat_col] = pd.cut(X_cls_df[cat_col], bins=4, labels=[f'A{i}', f'B{i}', f'C{i}', f'D{i}'])
    cls_df = X_cls_df.copy()
    cls_df['target'] = y_cls

    # Define feature types
    numerical_features = [f'num_{i}' for i in range(7)]
    categorical_features = [f'cat_{i}' for i in range(3)]

    # 2. Split data
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = split_data(cls_df, 'target')

    # 3. Create preprocessing pipeline
    # Note: Fit the pipeline ONLY on training data
    preprocessor_cls = create_preprocessing_pipeline(numerical_features, categorical_features)
    preprocessor_cls.fit(X_train_cls) # Fit the preprocessor

    # 4. Train model (using the fitted preprocessor within train_model)
    rf_cls = RandomForestClassifier(random_state=DEFAULT_RANDOM_STATE, n_estimators=50)
    # Pass the *unfitted* model and the *fitted* preprocessor
    trained_cls_pipeline = train_model(rf_cls, X_train_cls, y_train_cls, preprocessing_pipeline=preprocessor_cls)

    # 5. Evaluate
    # The trained_cls_pipeline now handles preprocessing internally
    cls_metrics = evaluate_classification(trained_cls_pipeline, X_test_cls, y_test_cls)

    # 6. Save and Load
    model_path_cls = "example_classification_model.joblib"
    save_model(trained_cls_pipeline, model_path_cls)
    loaded_cls_model = load_model(model_path_cls)

    # Verify loaded model prediction
    if loaded_cls_model:
        pred = loaded_cls_model.predict(X_test_cls[:5])
        logger.info(f"Prediction from loaded classification model (first 5): {pred}")


    # --- Regression Example ---
    logger.info("\n--- Regression Example ---")
    # 1. Create synthetic data
    X_reg, y_reg = make_regression(n_samples=200, n_features=5, n_informative=3,
                                   noise=10, random_state=DEFAULT_RANDOM_STATE)
    X_reg_df = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(5)])
    reg_df = X_reg_df.copy()
    reg_df['target'] = y_reg

    # Assume all features are numerical for simplicity here
    numerical_features_reg = list(X_reg_df.columns)
    categorical_features_reg = []

    # 2. Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(reg_df, 'target', stratify=False) # No stratification for regression

    # 3. Create preprocessing pipeline (only scaling in this case)
    preprocessor_reg = create_preprocessing_pipeline(numerical_features_reg, categorical_features_reg, scaler=StandardScaler())
    preprocessor_reg.fit(X_train_reg)

    # 4. Train model
    rf_reg = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, n_estimators=50)
    trained_reg_pipeline = train_model(rf_reg, X_train_reg, y_train_reg, preprocessing_pipeline=preprocessor_reg)

    # 5. Evaluate
    reg_metrics = evaluate_regression(trained_reg_pipeline, X_test_reg, y_test_reg)

    # 6. Save and Load
    model_path_reg = "example_regression_model.joblib"
    save_model(trained_reg_pipeline, model_path_reg)
    loaded_reg_model = load_model(model_path_reg)

    # Verify loaded model prediction
    if loaded_reg_model:
        pred = loaded_reg_model.predict(X_test_reg[:5])
        logger.info(f"Prediction from loaded regression model (first 5): {pred}")

    logger.info("\n--- Example Usage Finished ---")