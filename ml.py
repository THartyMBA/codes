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


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


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