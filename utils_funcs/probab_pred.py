import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import pickle
import json
import os
import warnings
import logging
from dataclasses import dataclass
from datetime import datetime
import joblib
from collections import defaultdict
import base64
from io import BytesIO
import re
import string
import time

# Machine learning libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, 
    LabelEncoder, OrdinalEncoder, PowerTransformer, QuantileTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, VarianceThreshold,
    f_classif, mutual_info_classif, chi2
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, VotingClassifier, StackingClassifier,
    IsolationForest
)
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, RidgeClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss, 
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning

# For XGBoost, LightGBM and CatBoost
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
try:
    import catboost as cb
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# For explanation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
try:
    from eli5 import show_weights, show_prediction
    from eli5.sklearn import PermutationImportance
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False

# For text features
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    import nltk
    NLTK_AVAILABLE = True
    # Ensure necessary nltk packages are downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    TEXT_VECTORIZERS_AVAILABLE = True
except ImportError:
    TEXT_VECTORIZERS_AVAILABLE = False

# For API deployment
try:
    import flask
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ProbabilityPrediction')

# Suppress some common warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('display.max_columns', None)


#================================================================
# Data Loading and Preprocessing Module
#================================================================

class DataLoader:
    """
    Class for loading and initial exploration of data
    """
    
    @staticmethod
    def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file
        
        Args:
            filepath: Path to the CSV file
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"Loaded data from {filepath}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    @staticmethod
    def load_excel(filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from an Excel file
        
        Args:
            filepath: Path to the Excel file
            **kwargs: Additional arguments to pass to pd.read_excel
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_excel(filepath, **kwargs)
            logger.info(f"Loaded data from {filepath}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
    
    @staticmethod
    def load_sql(query: str, connection_string: str) -> pd.DataFrame:
        """
        Load data from a SQL database
        
        Args:
            query: SQL query to execute
            connection_string: Database connection string
            
        Returns:
            DataFrame with loaded data
        """
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(connection_string)
            df = pd.read_sql(query, engine)
            logger.info(f"Loaded data from SQL: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading data from SQL: {e}")
            raise
    
    @staticmethod
    def get_data_info(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
        """
        Get basic information about the dataset
        
        Args:
            df: DataFrame to analyze
            verbose: Whether to print information
            
        Returns:
            Dictionary with data information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percent': (df.isnull().sum() / len(df) * 100).to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns},
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024)  # In MB
        }
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Try to detect date columns that are stored as strings
        potential_date_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > 0:
                try:
                    pd.to_datetime(df[col].dropna().iloc[0])
                    potential_date_cols.append(col)
                except:
                    pass
        
        info['column_types'] = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'date': date_cols,
            'potential_date': potential_date_cols
        }
        
        # Basic statistics for numeric columns
        if numeric_cols:
            info['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # Class distribution for potential target columns
        for col in df.columns:
            if df[col].nunique() <= 10:  # Potential categorical target
                info[f'{col}_distribution'] = df[col].value_counts().to_dict()
        
        if verbose:
            print(f"Dataset Shape: {info['shape']}")
            print(f"Memory Usage: {info['memory_usage']:.2f} MB")
            print("\nColumn Types:")
            for type_name, cols in info['column_types'].items():
                if cols:
                    print(f"  {type_name.capitalize()}: {', '.join(cols)}")
            
            print("\nMissing Values:")
            missing_df = pd.DataFrame({
                'Column': info['missing_values'].keys(),
                'Missing Values': info['missing_values'].values(),
                'Missing Percent': [f"{x:.2f}%" for x in info['missing_percent'].values()]
            }).sort_values('Missing Values', ascending=False)
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            if not missing_df.empty:
                print(missing_df.to_string(index=False))
            else:
                print("  No missing values")
        
        return info


class FeatureAnalyzer:
    """
    Class for analyzing features in the dataset
    """
    
    @staticmethod
    def analyze_numeric_features(df: pd.DataFrame, 
                                cols: List[str] = None,
                                plot: bool = True,
                                figsize: Tuple[int, int] = (15, 10)) -> Dict[str, Dict[str, float]]:
        """
        Analyze numeric features
        
        Args:
            df: DataFrame with data
            cols: List of numeric columns to analyze (if None, all numeric columns are used)
            plot: Whether to create visualization
            figsize: Figure size for plots
            
        Returns:
            Dictionary with analysis results
        """
        if cols is None:
            cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not cols:
            logger.warning("No numeric columns to analyze")
            return {}
        
        results = {}
        
        for col in cols:
            # Basic statistics
            stats = df[col].describe().to_dict()
            
            # Additional statistics
            stats['missing'] = df[col].isnull().sum()
            stats['missing_percent'] = df[col].isnull().sum() / len(df) * 100
            stats['zeros'] = (df[col] == 0).sum()
            stats['zeros_percent'] = (df[col] == 0).sum() / len(df) * 100
            
            # Skewness and kurtosis
            stats['skewness'] = df[col].skew()
            stats['kurtosis'] = df[col].kurtosis()
            
            # Check for outliers using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
            stats['outliers'] = len(outliers)
            stats['outliers_percent'] = len(outliers) / len(df) * 100
            
            results[col] = stats
        
        if plot and cols:
            # Create visualization
            n_cols = min(3, len(cols))
            n_rows = (len(cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, col in enumerate(cols):
                if i < len(axes):
                    # Histogram with KDE
                    sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
                    axes[i].set_title(f"{col}\nSkew: {results[col]['skewness']:.2f}, Kurt: {results[col]['kurtosis']:.2f}")
                    
                    # Add a boxplot at the bottom
                    boxax = axes[i].twinx()
                    boxax.boxplot(df[col].dropna(), vert=False, widths=0.7)
                    boxax.set(yticklabels=[])
                    boxax.set_ylabel('')
            
            for i in range(len(cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            # Create correlation heatmap if there are multiple columns
            if len(cols) > 1:
                plt.figure(figsize=(10, 8))
                corr = df[cols].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                
                sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                           fmt='.2f', linewidths=0.5)
                plt.title('Feature Correlation Heatmap')
                plt.tight_layout()
                plt.show()
        
        return results
    
    @staticmethod
    def analyze_categorical_features(df: pd.DataFrame, 
                                    cols: List[str] = None,
                                    plot: bool = True,
                                    figsize: Tuple[int, int] = (15, 10),
                                    max_categories: int = 20) -> Dict[str, Dict[str, Any]]:
        """
        Analyze categorical features
        
        Args:
            df: DataFrame with data
            cols: List of categorical columns to analyze (if None, all categorical columns are used)
            plot: Whether to create visualization
            figsize: Figure size for plots
            max_categories: Maximum number of categories to display in plots
            
        Returns:
            Dictionary with analysis results
        """
        if cols is None:
            cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        if not cols:
            logger.warning("No categorical columns to analyze")
            return {}
        
        results = {}
        
        for col in cols:
            # Basic statistics
            value_counts = df[col].value_counts()
            value_percent = df[col].value_counts(normalize=True) * 100
            
            stats = {
                'count': len(df[col]),
                'unique': df[col].nunique(),
                'missing': df[col].isnull().sum(),
                'missing_percent': df[col].isnull().sum() / len(df) * 100,
                'most_common': value_counts.index[0] if not value_counts.empty else None,
                'most_common_count': value_counts.iloc[0] if not value_counts.empty else None,
                'most_common_percent': value_percent.iloc[0] if not value_percent.empty else None,
                'value_counts': value_counts.to_dict(),
                'value_percent': value_percent.to_dict()
            }
            
            results[col] = stats
        
        if plot and cols:
            # Create visualization
            n_cols = min(2, len(cols))
            n_rows = (len(cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, col in enumerate(cols):
                if i < len(axes):
                    # Get value counts
                    vc = df[col].value_counts()
                    
                    # If too many categories, limit display
                    if len(vc) > max_categories:
                        top_categories = vc.head(max_categories - 1).index.tolist()
                        other_mask = ~df[col].isin(top_categories)
                        
                        # Create a copy to avoid SettingWithCopyWarning
                        plot_df = df.copy()
                        plot_df.loc[other_mask, col] = 'Other'
                        
                        # Recalculate value counts
                        vc = plot_df[col].value_counts()
                    
                    # Sort by value
                    vc = vc.sort_values(ascending=False)
                    
                    # Create the plot
                    sns.barplot(x=vc.index, y=vc.values, ax=axes[i])
                    axes[i].set_title(f"{col}\n(Unique: {results[col]['unique']})")
                    
                    # Rotate labels if there are more than 5 categories
                    if len(vc) > 5:
                        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
                    
                    axes[i].set_ylabel('Count')
                    
                    # Add count and percentage labels
                    for j, p in enumerate(axes[i].patches):
                        category = vc.index[j]
                        count = vc.values[j]
                        percent = count / len(df) * 100
                        
                        # If bar is too small, don't add text
                        if p.get_height() < vc.max() * 0.03:
                            continue
                        
                        axes[i].annotate(f'{count}\n({percent:.1f}%)', 
                                       (p.get_x() + p.get_width() / 2., p.get_height()),
                                       ha='center', va='bottom', fontsize=8)
            
            for i in range(len(cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        
        return results
    
    @staticmethod
    def analyze_target_relationship(df: pd.DataFrame, 
                                   target: str,
                                   features: List[str] = None,
                                   figsize: Tuple[int, int] = (15, 10),
                                   max_categories: int = 20) -> Dict[str, Dict[str, float]]:
        """
        Analyze relationship between features and target variable
        
        Args:
            df: DataFrame with data
            target: Target column name
            features: List of feature columns to analyze (if None, all columns except target are used)
            figsize: Figure size for plots
            max_categories: Maximum number of categories to display in plots
            
        Returns:
            Dictionary with analysis results
        """
        if features is None:
            features = [col for col in df.columns if col != target]
        
        if target not in df.columns:
            logger.error(f"Target column '{target}' not found in DataFrame")
            return {}
        
        # Check if target is binary (0/1 or Yes/No)
        unique_vals = df[target].dropna().unique()
        is_binary = len(unique_vals) == 2
        
        if is_binary:
            # If target is string-based, convert to 0/1
            if df[target].dtype == 'object' or df[target].dtype == 'bool':
                # Create mapping to convert to 0/1
                target_mapping = {val: i for i, val in enumerate(unique_vals)}
                target_series = df[target].map(target_mapping)
            else:
                target_series = df[target]
            
            logger.info(f"Binary target detected: {unique_vals}")
        else:
            target_series = df[target]
            logger.info(f"Non-binary target detected with {len(unique_vals)} unique values")
        
        results = {}
        
        # Analyze each feature
        for feature in features:
            if feature == target:
                continue
                
            # Skip if feature has all missing values
            if df[feature].isnull().sum() == len(df):
                continue
                
            # Get feature type
            is_numeric = pd.api.types.is_numeric_dtype(df[feature])
            
            if is_numeric:
                # For numeric features
                stats = {
                    'correlation': df[feature].corr(target_series),
                    'correlation_squared': df[feature].corr(target_series) ** 2,
                }
                
                # Calculate additional metrics
                if is_binary:
                    # Calculate metrics specific to binary classification
                    
                    # Point-biserial correlation
                    stats['point_biserial'] = df[feature].corr(target_series)
                    
                    # Feature means by class
                    stats['mean_class_0'] = df[df[target] == unique_vals[0]][feature].mean()
                    stats['mean_class_1'] = df[df[target] == unique_vals[1]][feature].mean()
                    
                    # Feature standard deviations by class
                    stats['std_class_0'] = df[df[target] == unique_vals[0]][feature].std()
                    stats['std_class_1'] = df[df[target] == unique_vals[1]][feature].std()
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt((stats['std_class_0']**2 + stats['std_class_1']**2) / 2)
                    if pooled_std > 0:
                        stats['effect_size'] = abs(stats['mean_class_1'] - stats['mean_class_0']) / pooled_std
                    else:
                        stats['effect_size'] = 0
                    
                    # Calculate AUC for the feature as a predictor
                    try:
                        from sklearn.metrics import roc_auc_score
                        # Handle case where higher values might indicate lower probability
                        auc = roc_auc_score(target_series, df[feature])
                        stats['auc'] = max(auc, 1 - auc)  # Always get AUC ≥ 0.5
                    except:
                        stats['auc'] = 0.5
                
                # Create visualization
                plt.figure(figsize=(8, 6))
                
                if is_binary:
                    # Plot distributions by class
                    for i, val in enumerate(unique_vals):
                        sns.kdeplot(df[df[target] == val][feature].dropna(), 
                                   label=f"{target}={val}", shade=True)
                    
                    plt.title(f"{feature} Distribution by Target Class\n" +
                             f"Correlation: {stats['correlation']:.3f}, AUC: {stats['auc']:.3f}")
                else:
                    # Scatter plot for continuous target
                    plt.scatter(df[feature], df[target], alpha=0.5)
                    
                    # Add trend line
                    from scipy import stats as scipy_stats
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                        df[feature].dropna(), df[target].dropna())
                    
                    x = np.array([df[feature].min(), df[feature].max()])
                    y = intercept + slope * x
                    plt.plot(x, y, 'r', label=f'y = {slope:.3f}x + {intercept:.3f}')
                    
                    plt.title(f"{feature} vs {target}\n" +
                             f"Correlation: {stats['correlation']:.3f}, R²: {r_value**2:.3f}")
                
                plt.xlabel(feature)
                plt.ylabel(target)
                plt.legend()
                plt.tight_layout()
                plt.show()
            
            else:
                # For categorical features
                # Calculate statistics
                stats = {
                    'unique_values': df[feature].nunique(),
                }
                
                # Create a cross-tabulation
                crosstab = pd.crosstab(df[feature], df[target], normalize='index') * 100
                stats['crosstab'] = crosstab.to_dict()
                
                if is_binary:
                    # Calculate metrics for binary classification
                    
                    # Chi-squared test for independence
                    from scipy.stats import chi2_contingency
                    contingency_table = pd.crosstab(df[feature], df[target])
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    
                    stats['chi2'] = chi2
                    stats['p_value'] = p
                    stats['cramers_v'] = np.sqrt(chi2 / (len(df) * min(contingency_table.shape[0]-1, contingency_table.shape[1]-1)))
                    
                    # Information Value (IV) calculation
                    iv = 0
                    woe_dict = {}
                    
                    for category in df[feature].dropna().unique():
                        category_df = df[df[feature] == category]
                        non_event = (category_df[target] == unique_vals[0]).sum()
                        event = (category_df[target] == unique_vals[1]).sum()
                        
                        # Handle zero counts
                        if non_event == 0 or event == 0:
                            continue
                            
                        # Calculate proportions
                        total_non_event = (df[target] == unique_vals[0]).sum()
                        total_event = (df[target] == unique_vals[1]).sum()
                        
                        p_non_event = non_event / total_non_event
                        p_event = event / total_event
                        
                        # Weight of Evidence
                        woe = np.log(p_non_event / p_event)
                        woe_dict[category] = woe
                        
                        # Information Value
                        iv += (p_non_event - p_event) * woe
                    
                    stats['iv'] = abs(iv)
                    stats['woe'] = woe_dict
                
                # Create visualization
                # Limit categories if too many
                if df[feature].nunique() > max_categories:
                    # Get the most common categories
                    top_categories = df[feature].value_counts().head(max_categories - 1).index.tolist()
                    
                    # Create a copy to avoid SettingWithCopyWarning
                    plot_df = df.copy()
                    plot_df.loc[~plot_df[feature].isin(top_categories), feature] = 'Other'
                else:
                    plot_df = df
                
                plt.figure(figsize=(10, 6))
                
                if is_binary:
                    # Stacked bar chart for binary target
                    pd.crosstab(plot_df[feature], plot_df[target], normalize='index').plot(
                        kind='bar', stacked=True, colormap='coolwarm')
                    
                    plt.title(f"{feature} vs {target}\n" +
                             f"Cramer's V: {stats['cramers_v']:.3f}, IV: {stats.get('iv', 0):.3f}")
                    plt.ylabel("Proportion")
                else:
                    # Box plot for continuous target
                    sns.boxplot(x=feature, y=target, data=plot_df)
                    plt.title(f"{feature} vs {target}")
                
                plt.xlabel(feature)
                if df[feature].nunique() > 5:
                    plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
            
            results[feature] = stats
        
        return results


class DataPreprocessor:
    """
    Class for preprocessing data
    """
    
    def __init__(self):
        self.transformers = {}
        self.feature_names = None
    
    def detect_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect feature types from DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with feature types
        """
        feature_types = {
            'numeric': [],
            'categorical': [],
            'text': [],
            'datetime': [],
            'boolean': [],
            'id': [],
            'unknown': []
        }
        
        for col in df.columns:
            # Check if column is datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_types['datetime'].append(col)
                continue
            
            # Check if column is boolean
            if pd.api.types.is_bool_dtype(df[col]) or set(df[col].dropna().unique()).issubset({0, 1, True, False}):
                feature_types['boolean'].append(col)
                continue
            
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's likely an ID column
                if df[col].nunique() == df.shape[0] or (col.lower().endswith('id') and df[col].nunique() > 0.9 * df.shape[0]):
                    feature_types['id'].append(col)
                else:
                    feature_types['numeric'].append(col)
                continue
            
            # Check if it's categorical or text
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                # Calculate statistics to determine text vs categorical
                val_lens = df[col].dropna().astype(str).str.len()
                
                if (val_lens.mean() > 50 or  # Long values suggest text
                    df[col].nunique() > 0.9 * df[col].count()):  # High cardinality suggests text or ID
                    feature_types['text'].append(col)
                else:
                    feature_types['categorical'].append(col)
                continue
            
            # If we get here, it's an unknown type
            feature_types['unknown'].append(col)
        
        return feature_types
    
    def fit_transform(self, df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Preprocess data based on configuration
        
        Args:
            df: Input DataFrame
            config: Dictionary with preprocessing configuration
                   (if None, auto-detect and apply default preprocessing)
            
        Returns:
            Preprocessed DataFrame
        """
        df_copy = df.copy()
        
        if config is None:
            # Auto-detect feature types and create default config
            feature_types = self.detect_feature_types(df_copy)
            
            config = {
                'numeric': {
                    'columns': feature_types['numeric'],
                    'missing_strategy': 'median',
                    'outlier_strategy': 'clip',
                    'scaling': 'standard'
                },
                'categorical': {
                    'columns': feature_types['categorical'],
                    'missing_strategy': 'most_frequent',
                    'encoding': 'onehot',
                    'max_categories': 10
                },
                'text': {
                    'columns': feature_types['text'],
                    'vectorizer': 'tfidf',
                    'max_features': 100
                },
                'datetime': {
                    'columns': feature_types['datetime'],
                    'extract_components': True
                },
                'drop': feature_types['id'] + feature_types['unknown']
            }
        
        # Process each feature type
        result_dfs = []
        
        # Numeric features
        if 'numeric' in config and config['numeric']['columns']:
            numeric_cols = config['numeric']['columns']
            numeric_df = df_copy[numeric_cols].copy()
            
            # Handle missing values
            if 'missing_strategy' in config['numeric']:
                strategy = config['numeric']['missing_strategy']
                if strategy == 'drop':
                    df_copy = df_copy.dropna(subset=numeric_cols)
                    numeric_df = df_copy[numeric_cols].copy()
                elif strategy in ['mean', 'median', 'most_frequent']:
                    imputer = SimpleImputer(strategy=strategy)
                    numeric_df = pd.DataFrame(
                        imputer.fit_transform(numeric_df),
                        columns=numeric_cols,
                        index=df_copy.index
                    )
                    self.transformers['numeric_imputer'] = imputer
            
            # Handle outliers
            if 'outlier_strategy' in config['numeric']:
                strategy = config['outlier_strategy'] if 'outlier_strategy' in config['numeric'] else None
                if strategy == 'clip':
                    for col in numeric_cols:
                        Q1 = numeric_df[col].quantile(0.25)
                        Q3 = numeric_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        numeric_df[col] = numeric_df[col].clip(lower_bound, upper_bound)
            
            # Apply scaling
            if 'scaling' in config['numeric']:
                scaling = config['numeric']['scaling']
                if scaling == 'standard':
                    scaler = StandardScaler()
                elif scaling == 'minmax':
                    scaler = MinMaxScaler()
                elif scaling == 'robust':
                    scaler = RobustScaler()
                else:
                    scaler = None
                
                if scaler:
                    numeric_df = pd.DataFrame(
                        scaler.fit_transform(numeric_df),
                        columns=numeric_cols,
                        index=df_copy.index
                    )
                    self.transformers['numeric_scaler'] = scaler
            
            result_dfs.append(numeric_df)
        
        # Categorical features
        if 'categorical' in config and config['categorical']['columns']:
            cat_cols = config['categorical']['columns']
            cat_df = df_copy[cat_cols].copy()
            
            # Handle missing values
            if 'missing_strategy' in config['categorical']:
                strategy = config['categorical']['missing_strategy']
                if strategy == 'drop':
                    df_copy = df_copy.dropna(subset=cat_cols)
                    cat_df = df_copy[cat_cols].copy()
                elif strategy in ['most_frequent']:
                    imputer = SimpleImputer(strategy=strategy)
                    cat_df = pd.DataFrame(
                        imputer.fit_transform(cat_df),
                        columns=cat_cols,
                        index=df_copy.index
                    )
                    self.transformers['categorical_imputer'] = imputer
                elif strategy == 'new_category':
                    for col in cat_cols:
                        cat_df[col] = cat_df[col].fillna('Missing')
            
            # Apply encoding
            if 'encoding' in config['categorical']:
                encoding = config['categorical']['encoding']
                if encoding == 'onehot':
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore',
                                          max_categories=config['categorical'].get('max_categories', None))
                    encoded_array = encoder.fit_transform(cat_df)
                    
                    # Get feature names and create DataFrame
                    feature_names = encoder.get_feature_names_out(cat_cols)
                    encoded_df = pd.DataFrame(
                        encoded_array, 
                        columns=feature_names,
                        index=df_copy.index
                    )
                    self.transformers['categorical_encoder'] = encoder
                    
                    result_dfs.append(encoded_df)
                elif encoding == 'label':
                    # Label encoding for each column
                    for col in cat_cols:
                        encoder = LabelEncoder()
                        cat_df[col] = encoder.fit_transform(cat_df[col])
                        self.transformers[f'label_encoder_{col}'] = encoder
                    
                    result_dfs.append(cat_df)
                elif encoding == 'target':
                    # Target encoding needs target variable
                    if 'target_column' in config['categorical']:
                        target_col = config['categorical']['target_column']
                        target_vals = df_copy[target_col]
                        
                        for col in cat_cols:
                            # Calculate mean target value for each category
                            target_means = df_copy.groupby(col)[target_col].mean()
                            global_mean = df_copy[target_col].mean()
                            
                            # Apply smoothing
                            smoothing = config['categorical'].get('smoothing', 10)
                            counts = df_copy.groupby(col)[target_col].count()
                            
                            # Smoothed target encoding
                            smooth_means = (counts * target_means + smoothing * global_mean) / (counts + smoothing)
                            cat_df[col] = cat_df[col].map(smooth_means)
                            
                            # Store encoding mapping
                            self.transformers[f'target_encoder_{col}'] = smooth_means.to_dict()
                        
                        result_dfs.append(cat_df)
            
        # Text features
        if 'text' in config and config['text']['columns'] and TEXT_VECTORIZERS_AVAILABLE:
            text_cols = config['text']['columns']
            
            for col in text_cols:
                # Clean text
                df_copy[col] = df_copy[col].fillna('').astype(str)
                
                # Apply vectorization
                vectorizer_type = config['text'].get('vectorizer', 'tfidf')
                max_features = config['text'].get('max_features', 100)
                
                if vectorizer_type == 'tfidf':
                    vectorizer = TfidfVectorizer(
                        max_features=max_features,
                        stop_words='english',
                        min_df=5,
                        max_df=0.8
                    )
                elif vectorizer_type == 'count':
                    vectorizer = CountVectorizer(
                        max_features=max_features,
                        stop_words='english',
                        min_df=5,
                        max_df=0.8
                    )
                
                # Apply vectorizer to column
                text_features = vectorizer.fit_transform(df_copy[col])
                
                # Convert to DataFrame with proper feature names
                feature_names = [f"{col}_{f}" for f in vectorizer.get_feature_names_out()]
                text_df = pd.DataFrame(
                    text_features.toarray(),
                    columns=feature_names,
                    index=df_copy.index
                )
                
                self.transformers[f'text_vectorizer_{col}'] = vectorizer
                result_dfs.append(text_df)
        
        # Datetime features
        if 'datetime' in config and config['datetime']['columns']:
            datetime_cols = config['datetime']['columns']
            
            for col in datetime_cols:
                if not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    try:
                        df_copy[col] = pd.to_datetime(df_copy[col])
                    except:
                        logger.warning(f"Could not convert {col} to datetime, skipping")
                        continue
                
                if config['datetime'].get('extract_components', True):
                    dt_df = pd.DataFrame(index=df_copy.index)
                    dt_df[f'{col}_year'] = df_copy[col].dt.year
                    dt_df[f'{col}_month'] = df_copy[col].dt.month
                    dt_df[f'{col}_day'] = df_copy[col].dt.day
                    dt_df[f'{col}_dayofweek'] = df_copy[col].dt.dayofweek
                    
                    if df_copy[col].dt.hour.nunique() > 1:  # If time components vary
                        dt_df[f'{col}_hour'] = df_copy[col].dt.hour
                        dt_df[f'{col}_minute'] = df_copy[col].dt.minute
                    
                    result_dfs.append(dt_df)
        
        # Combine all processed DataFrames
        if result_dfs:
            processed_df = pd.concat(result_dfs, axis=1)
            self.feature_names = processed_df.columns.tolist()
            return processed_df
        else:
            logger.warning("No features were processed, returning original DataFrame")
            return df_copy
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply pre-fitted transformations to new data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.transformers:
            logger.error("No transformers found. Call fit_transform first.")
            return df
        
        df_copy = df.copy()
        result_dfs = []
        
        # Apply numeric transformations
        if 'numeric_imputer' in self.transformers:
            numeric_cols = self.transformers['numeric_imputer'].feature_names_in_
            if all(col in df_copy.columns for col in numeric_cols):
                numeric_df = df_copy[numeric_cols].copy()
                numeric_df = pd.DataFrame(
                    self.transformers['numeric_imputer'].transform(numeric_df),
                    columns=numeric_cols,
                    index=df_copy.index
                )
                
                if 'numeric_scaler' in self.transformers:
                    numeric_df = pd.DataFrame(
                        self.transformers['numeric_scaler'].transform(numeric_df),
                        columns=numeric_cols,
                        index=df_copy.index
                    )
                
                result_dfs.append(numeric_df)
        
        # Apply categorical transformations
        if 'categorical_encoder' in self.transformers:
            encoder = self.transformers['categorical_encoder']
            cat_cols = encoder.feature_names_in_
            
            if all(col in df_copy.columns for col in cat_cols):
                cat_df = df_copy[cat_cols].copy()
                
                if 'categorical_imputer' in self.transformers:
                    cat_df = pd.DataFrame(
                        self.transformers['categorical_imputer'].transform(cat_df),
                        columns=cat_cols,
                        index=df_copy.index
                    )
                
                encoded_array = encoder.transform(cat_df)
                feature_names = encoder.get_feature_names_out(cat_cols)
                encoded_df = pd.DataFrame(
                    encoded_array, 
                    columns=feature_names,
                    index=df_copy.index
                )
                
                result_dfs.append(encoded_df)
        
        # Apply label encoding transformations
        for col in df_copy.columns:
            key = f'label_encoder_{col}'
            if key in self.transformers:
                encoder = self.transformers[key]
                # Create a copy to avoid modifying the original
                encoded_col = df_copy[[col]].copy()
                
                # Handle unseen categories
                encoded_col[col] = encoded_col[col].apply(
                    lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                )
                
                # Transform
                encoded_col[col] = encoder.transform(encoded_col[col])
                result_dfs.append(encoded_col)
        
        # Apply target encoding transformations
        for col in df_copy.columns:
            key = f'target_encoder_{col}'
            if key in self.transformers:
                mapping = self.transformers[key]
                # Create a copy to avoid modifying the original
                encoded_col = pd.DataFrame(index=df_copy.index)
                
                # Apply mapping with fallback to global mean for unseen categories
                global_mean = np.mean(list(mapping.values()))
                encoded_col[col] = df_copy[col].map(mapping).fillna(global_mean)
                
                result_dfs.append(encoded_col)
        
        # Apply text vectorization transformations
        for key, vectorizer in self.transformers.items():
            if key.startswith('text_vectorizer_'):
                col = key.replace('text_vectorizer_', '')
                if col in df_copy.columns:
                    # Clean text
                    text_series = df_copy[col].fillna('').astype(str)
                    
                    # Apply vectorizer
                    text_features = vectorizer.transform(text_series)
                    
                    # Convert to DataFrame with proper feature names
                    feature_names = [f"{col}_{f}" for f in vectorizer.get_feature_names_out()]
                    text_df = pd.DataFrame(
                        text_features.toarray(),
                        columns=feature_names,
                        index=df_copy.index
                    )
                    
                    result_dfs.append(text_df)
        
        # Process datetime columns
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                dt_df = pd.DataFrame(index=df_copy.index)
                dt_df[f'{col}_year'] = df_copy[col].dt.year
                dt_df[f'{col}_month'] = df_copy[col].dt.month
                dt_df[f'{col}_day'] = df_copy[col].dt.day
                dt_df[f'{col}_dayofweek'] = df_copy[col].dt.dayofweek
                
                if df_copy[col].dt.hour.nunique() > 1:  # If time components vary
                    dt_df[f'{col}_hour'] = df_copy[col].dt.hour
                    dt_df[f'{col}_minute'] = df_copy[col].dt.minute
                
                result_dfs.append(dt_df)
        
        # Combine all processed DataFrames
        if result_dfs:
            processed_df = pd.concat(result_dfs, axis=1)
            
            # Ensure all expected features are present, add missing ones with zeros
            if self.feature_names:
                missing_cols = set(self.feature_names) - set(processed_df.columns)
                if missing_cols:
                    for col in missing_cols:
                        processed_df[col] = 0
                
                # Reorder columns to match training data
                processed_df = processed_df[self.feature_names]
            
            return processed_df
        else:
            logger.warning("No features were transformed, returning original DataFrame")
            return df_copy
    
    def save(self, filepath: str) -> None:
        """
        Save preprocessor to file
        
        Args:
            filepath: Path to save the preprocessor
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'DataPreprocessor':
        """
        Load preprocessor from file
        
        Args:
            filepath: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor
        """
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


# =============================================================================
# Feature Engineering Module
# =============================================================================

class FeatureGenerator:
    """
    Class for generating features
    """
    
    @staticmethod
    def create_polynomial_features(df: pd.DataFrame, 
                                  columns: List[str], 
                                  degree: int = 2,
                                  interaction_only: bool = False) -> pd.DataFrame:
        """
        Create polynomial features from numeric columns
        
        Args:
            df: Input DataFrame
            columns: List of columns to use
            degree: Polynomial degree
            interaction_only: Whether to only include interaction terms
            
        Returns:
            DataFrame with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        result = df.copy()
        
        if not columns:
            return result
        
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        poly_features = poly.fit_transform(result[columns])
        
        # Get feature names
        feature_names = poly.get_feature_names_out(columns)
        
        # Add polynomial features to DataFrame
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=result.index)
        
        # Drop original columns to avoid duplication
        poly_df = poly_df.drop(columns, axis=1, errors='ignore')
        
        return pd.concat([result, poly_df], axis=1)
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame, 
                                   feature_pairs: List[Tuple[str, str]], 
                                   operations: List[str] = ['multiply']) -> pd.DataFrame:
        """
        Create interaction features between pairs of columns
        
        Args:
            df: Input DataFrame
            feature_pairs: List of (column1, column2) pairs
            operations: List of operations to apply ('multiply', 'divide', 'add', 'subtract')
            
        Returns:
            DataFrame with interaction features
        """
        result = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                logger.warning(f"Columns {col1} or {col2} not found, skipping")
                continue
            
            for op in operations:
                if op == 'multiply':
                    result[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                elif op == 'divide':
                    # Handle division by zero
                    result[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                elif op == 'add':
                    result[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                elif op == 'subtract':
                    result[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
        
        return result
    
    @staticmethod
    def create_binned_features(df: pd.DataFrame, 
                              columns: List[str], 
                              n_bins: int = 5,
                              strategy: str = 'quantile') -> pd.DataFrame:
        """
        Create binned features from numeric columns
        
        Args:
            df: Input DataFrame
            columns: List of columns to bin
            n_bins: Number of bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            
        Returns:
            DataFrame with binned features
        """
        from sklearn.preprocessing import KBinsDiscretizer
        
        result = df.copy()
        
        if not columns:
            return result
        
        # Dictionary to store discretizers
        discretizers = {}
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            # Skip columns with too many missing values
            if df[col].isnull().sum() > 0.5 * len(df):
                logger.warning(f"Column {col} has too many missing values, skipping")
                continue
            
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
            
            # Handle NaN values
            not_null = ~df[col].isnull()
            if not_null.sum() > 0:
                # Fit and transform non-null values
                binned_values = discretizer.fit_transform(df.loc[not_null, [col]])
                
                # Create new column for binned values
                result[f'{col}_bin'] = np.nan
                result.loc[not_null, f'{col}_bin'] = binned_values
                
                # Store discretizer
                discretizers[col] = discretizer
        
        # Add as attributes to the returned DataFrame
        result.discretizers = discretizers
        
        return result
    
    @staticmethod
    def create_clustering_features(df: pd.DataFrame, 
                                  columns: List[str], 
                                  n_clusters: int = 3,
                                  algorithm: str = 'kmeans') -> pd.DataFrame:
        """
        Create clustering features from numeric columns
        
        Args:
            df: Input DataFrame
            columns: List of columns to use for clustering
            n_clusters: Number of clusters
            algorithm: Clustering algorithm ('kmeans', 'hierarchical', 'dbscan')
            
        Returns:
            DataFrame with cluster labels
        """
        result = df.copy()
        
        if not columns or not all(col in df.columns for col in columns):
            logger.warning("Some columns not found, skipping clustering")
            return result
        
        # Extract the features for clustering
        X = df[columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Normalize the features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply clustering
        if algorithm == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif algorithm == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif algorithm == 'dbscan':
            from sklearn.cluster import DBSCAN
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            logger.error(f"Unknown clustering algorithm: {algorithm}")
            return result
        
        # Fit and predict cluster labels
        cluster_labels = clusterer.fit_predict(X_scaled)
        
        # Add cluster labels to the result DataFrame
        result[f'cluster_{algorithm}'] = cluster_labels
        
        # Add clusterer as an attribute to the returned DataFrame
        result.clusterer = clusterer
        result.cluster_scaler = scaler
        
        return result
    
    @staticmethod
    def create_aggregation_features(df: pd.DataFrame, 
                                   group_column: str, 
                                   agg_columns: List[str],
                                   agg_funcs: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
        Create aggregation features based on a grouping column
        
        Args:
            df: Input DataFrame
            group_column: Column to group by
            agg_columns: Columns to aggregate
            agg_funcs: Aggregation functions to apply
            
        Returns:
            DataFrame with aggregation features
        """
        result = df.copy()
        
        if group_column not in df.columns:
            logger.warning(f"Group column {group_column} not found, skipping")
            return result
        
        agg_cols = [col for col in agg_columns if col in df.columns]
        if not agg_cols:
            logger.warning("No valid aggregation columns found, skipping")
            return result
        
        # Calculate aggregations
        for col in agg_cols:
            for func in agg_funcs:
                # Skip if column is not numeric
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                # Calculate aggregation
                agg_values = df.groupby(group_column)[col].agg(func)
                
                # Add as new feature
                result[f'{col}_{func}_by_{group_column}'] = df[group_column].map(agg_values)
        
        return result
    
    @staticmethod
    def create_date_features(df: pd.DataFrame, 
                            date_columns: List[str]) -> pd.DataFrame:
        """
        Create features from date columns
        
        Args:
            df: Input DataFrame
            date_columns: List of date columns
            
        Returns:
            DataFrame with date features
        """
        result = df.copy()
        
        for col in date_columns:
            if col not in df.columns:
                logger.warning(f"Date column {col} not found, skipping")
                continue
            
            # Ensure column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    result[col] = pd.to_datetime(df[col])
                except:
                    logger.warning(f"Could not convert {col} to datetime, skipping")
                    continue
            
            # Extract date components
            result[f'{col}_year'] = result[col].dt.year
            result[f'{col}_month'] = result[col].dt.month
            result[f'{col}_day'] = result[col].dt.day
            result[f'{col}_dayofweek'] = result[col].dt.dayofweek
            result[f'{col}_dayofyear'] = result[col].dt.dayofyear
            result[f'{col}_quarter'] = result[col].dt.quarter
            result[f'{col}_is_month_start'] = result[col].dt.is_month_start.astype(int)
            result[f'{col}_is_month_end'] = result[col].dt.is_month_end.astype(int)
            
            # Check if time components vary
            if result[col].dt.hour.nunique() > 1:
                result[f'{col}_hour'] = result[col].dt.hour
                result[f'{col}_minute'] = result[col].dt.minute
                
            # Create cyclical features for month, day of week
            result[f'{col}_month_sin'] = np.sin(2 * np.pi * result[col].dt.month / 12)
            result[f'{col}_month_cos'] = np.cos(2 * np.pi * result[col].dt.month / 12)
            result[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * result[col].dt.dayofweek / 7)
            result[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * result[col].dt.dayofweek / 7)
        
        return result
    
    @staticmethod
    def create_text_features(df: pd.DataFrame, 
                            text_columns: List[str]) -> pd.DataFrame:
        """
        Create features from text columns
        
        Args:
            df: Input DataFrame
            text_columns: List of text columns
            
        Returns:
            DataFrame with text features
        """
        result = df.copy()
        
        for col in text_columns:
            if col not in df.columns:
                logger.warning(f"Text column {col} not found, skipping")
                continue
            
            # Convert to string and handle missing values
            text_series = df[col].fillna('').astype(str)
            
            # Skip if all values are empty
            if text_series.str.strip().str.len().sum() == 0:
                logger.warning(f"Text column {col} is empty, skipping")
                continue
            
            # Basic text features
            result[f'{col}_length'] = text_series.str.len()
            result[f'{col}_word_count'] = text_series.str.split().str.len()
            result[f'{col}_upper_char_count'] = text_series.str.count(r'[A-Z]')
            result[f'{col}_lower_char_count'] = text_series.str.count(r'[a-z]')
            result[f'{col}_digit_count'] = text_series.str.count(r'[0-9]')
            result[f'{col}_special_char_count'] = text_series.str.count(r'[^\w\s]')
            result[f'{col}_has_url'] = text_series.str.contains('http|www').astype(int)
            
            # Advanced features if NLTK is available
            if NLTK_AVAILABLE:
                # Sentiment analysis with a simple approach
                import nltk
                from nltk.sentiment import SentimentIntensityAnalyzer
                
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except LookupError:
                    nltk.download('vader_lexicon')
                
                sia = SentimentIntensityAnalyzer()
                sentiments = text_series.apply(lambda x: sia.polarity_scores(x) if x else {})
                result[f'{col}_sentiment_neg'] = sentiments.apply(lambda x: x.get('neg', 0) if x else 0)
                result[f'{col}_sentiment_neu'] = sentiments.apply(lambda x: x.get('neu', 0) if x else 0)
                result[f'{col}_sentiment_pos'] = sentiments.apply(lambda x: x.get('pos', 0) if x else 0)
                result[f'{col}_sentiment_compound'] = sentiments.apply(lambda x: x.get('compound', 0) if x else 0)
        
        return result


class FeatureSelector:
    """
    Class for selecting features
    """
    
    def __init__(self):
        self.selected_features = None
        self.importance = None
        self.selector = None
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'rfe', 
                       n_features: int = None,
                       threshold: float = None,
                       **kwargs) -> pd.DataFrame:
        """
        Select features using various methods
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Feature selection method
            n_features: Number of features to select (used by some methods)
            threshold: Importance threshold for feature selection (used by some methods)
            **kwargs: Additional arguments for the selector
            
        Returns:
            DataFrame with selected features
        """
        if method == 'variance':
            # Variance threshold
            selector = VarianceThreshold(threshold=threshold or 0.01)
            X_selected = selector.fit_transform(X)
            self.selected_features = X.columns[selector.get_support()]
            self.selector = selector
            
        elif method == 'kbest':
            # SelectKBest with appropriate score function
            if n_features is None:
                n_features = min(10, X.shape[1])
            
            if y.nunique() <= 2:  # Binary classification
                score_func = kwargs.get('score_func', f_classif)
            else:  # Multi-class or regression
                score_func = kwargs.get('score_func', f_classif)
                
            selector = SelectKBest(score_func=score_func, k=n_features)
            X_selected = selector.fit_transform(X, y)
            self.selected_features = X.columns[selector.get_support()]
            self.importance = dict(zip(X.columns, selector.scores_))
            self.selector = selector
            
        elif method == 'rfe':
            # Recursive Feature Elimination
            if n_features is None:
                n_features = min(10, X.shape[1])
            
            # Choose estimator based on target type
            if y.nunique() <= 2:  # Binary classification
                estimator = kwargs.get('estimator', LogisticRegression(max_iter=1000, C=0.1))
            else:  # Multi-class or regression
                estimator = kwargs.get('estimator', RandomForestClassifier(n_estimators=100))
                
            selector = RFE(estimator=estimator, n_features_to_select=n_features)
            X_selected = selector.fit_transform(X, y)
            self.selected_features = X.columns[selector.get_support()]
            self.selector = selector
            
        elif method == 'model':
            # Select features based on model importance
            
            # Choose estimator based on target type
            if y.nunique() <= 2:  # Binary classification
                estimator = kwargs.get('estimator', RandomForestClassifier(n_estimators=100))
            else:  # Multi-class
                estimator = kwargs.get('estimator', RandomForestClassifier(n_estimators=100))
            
            # Fit the model
            estimator.fit(X, y)
            
            # Get feature importance
            if hasattr(estimator, 'feature_importances_'):
                importance = estimator.feature_importances_
            elif hasattr(estimator, 'coef_'):
                importance = np.abs(estimator.coef_)[0]
            else:
                raise ValueError("Estimator does not provide feature importance")
            
            self.importance = dict(zip(X.columns, importance))
            
            # Select features based on importance threshold or top n features
            if threshold is not None:
                selector = SelectFromModel(estimator, threshold=threshold)
            else:
                selector = SelectFromModel(estimator, max_features=n_features or 10)
                
            X_selected = selector.fit_transform(X, y)
            self.selected_features = X.columns[selector.get_support()]
            self.selector = selector
            
        elif method == 'lasso':
            # Lasso for feature selection
            from sklearn.linear_model import LassoCV
            
            # Default alpha values if not provided
            alphas = kwargs.get('alphas', [0.001, 0.01, 0.1, 1, 10, 100])
            
            # Initialize and fit LassoCV
            lasso = LassoCV(alphas=alphas, cv=5, max_iter=1000)
            lasso.fit(X, y)
            
            # Get coefficients
            coef = pd.Series(lasso.coef_, index=X.columns)
            self.importance = dict(zip(X.columns, np.abs(coef)))
            
            # Select non-zero coefficients
            self.selected_features = coef[coef != 0].index.tolist()
            
            if len(self.selected_features) == 0:
                logger.warning("Lasso selected 0 features. Using top 10 by absolute coefficient.")
                self.selected_features = coef.abs().sort_values(ascending=False).head(10).index.tolist()
            
            X_selected = X[self.selected_features]
            
        elif method == 'mutual_info':
            # Mutual information
            if n_features is None:
                n_features = min(10, X.shape[1])
            
            if y.nunique() <= 2:  # Binary classification
                mi = mutual_info_classif(X, y)
            else:  # Multi-class
                mi = mutual_info_classif(X, y)
                
            self.importance = dict(zip(X.columns, mi))
            
            # Select top n features by mutual information
            selected_idx = np.argsort(mi)[-n_features:]
            self.selected_features = X.columns[selected_idx]
            X_selected = X[self.selected_features]
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        return X[self.selected_features]
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature selection to new data
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with selected features
        """
        if self.selected_features is None:
            logger.error("No features selected. Call select_features first.")
            return X
        
        # Handle case when some selected features are missing in the new data
        missing_features = set(self.selected_features) - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features in the input data: {missing_features}")
            # Create DataFrame with only available selected features
            available_features = [f for f in self.selected_features if f in X.columns]
            return X[available_features]
        
        return X[self.selected_features]
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to display
            figsize: Figure size
        """
        if self.importance is None:
            logger.error("No feature importance available.")
            return
        
        # Convert to Series and sort
        importance = pd.Series(self.importance).sort_values(ascending=False)
        
        # Limit to top_n features
        if len(importance) > top_n:
            importance = importance.head(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(x=importance.values, y=importance.index)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()


# =============================================================================
# Model Building Module
# =============================================================================

class ProbabilityModel:
    """
    Base class for probability prediction models
    """
    
    def __init__(self, name: str = 'Base Model'):
        self.name = name
        self.model = None
        self.classes_ = None
        self.feature_names = None
        self.feature_importance_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ProbabilityModel':
        """
        Fit the model
        
        Args:
            X: Feature DataFrame
            y: Target Series
            **kwargs: Additional arguments for the model
            
        Returns:
            Self
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of positive class
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probabilities for each class
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class based on probability threshold
        
        Args:
            X: Feature DataFrame
            threshold: Probability threshold for positive class
            
        Returns:
            Array of predicted classes
        """
        probas = self.predict_proba(X)
        if probas.shape[1] == 2:  # Binary classification
            return (probas[:, 1] >= threshold).astype(int)
        else:  # Multi-class
            return np.argmax(probas, axis=1)
    
    def feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance if available
        
        Returns:
            Dictionary of feature importance
        """
        if self.feature_importance_ is not None:
            if isinstance(self.feature_importance_, np.ndarray):
                return dict(zip(self.feature_names, self.feature_importance_))
            return self.feature_importance_
        
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            if len(self.model.coef_.shape) == 1:
                return dict(zip(self.feature_names, np.abs(self.model.coef_)))
            else:
                return dict(zip(self.feature_names, np.mean(np.abs(self.model.coef_), axis=0)))
        else:
            logger.warning("Feature importance not available for this model.")
            return {}
    
    def save(self, filepath: str) -> None:
        """
        Save model to file
        
        Args:
            filepath: Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'ProbabilityModel':
        """
        Load model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model


class LogisticRegressionModel(ProbabilityModel):
    """
    Logistic Regression model for probability prediction
    """
    
    def __init__(self, C: float = 1.0, penalty: str = 'l2', solver: str = 'liblinear',
                class_weight: Union[str, Dict] = None, max_iter: int = 1000, 
                name: str = 'Logistic Regression'):
        """
        Initialize Logistic Regression model
        
        Args:
            C: Inverse of regularization strength
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
            solver: Algorithm to use in the optimization problem
            class_weight: Class weights (None, 'balanced', or dictionary)
            max_iter: Maximum number of iterations
            name: Model name
        """
        super().__init__(name)
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.class_weight = class_weight
        self.max_iter = max_iter
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'LogisticRegressionModel':
        """
        Fit Logistic Regression model
        
        Args:
            X: Feature DataFrame
            y: Target Series
            **kwargs: Additional arguments for LogisticRegression
            
        Returns:
            Self
        """
        self.feature_names = X.columns.tolist()
        self.classes_ = np.unique(y)
        
        # Create and fit the model
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            **kwargs
        )
        
        self.model.fit(X, y)
        
        # Store feature importance
        if hasattr(self.model, 'coef_'):
            if len(self.model.coef_.shape) == 1:
                self.feature_importance_ = np.abs(self.model.coef_)
            else:
                self.feature_importance_ = np.mean(np.abs(self.model.coef_), axis=0)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of each class
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probabilities for each class
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit first.")
        
        return self.model.predict_proba(X)


class RandomForestModel(ProbabilityModel):
    """
    Random Forest model for probability prediction
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None, 
                min_samples_split: int = 2, min_samples_leaf: int = 1,
                max_features: str = 'sqrt', class_weight: Union[str, Dict] = None,
                random_state: int = 42, name: str = 'Random Forest'):
        """
        Initialize Random Forest model
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            class_weight: Class weights (None, 'balanced', 'balanced_subsample', or dictionary)
            random_state: Random seed for reproducibility
            name: Model name
        """
        super().__init__(name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'RandomForestModel':
        """
        Fit Random Forest model
        
        Args:
            X: Feature DataFrame
            y: Target Series
            **kwargs: Additional arguments for RandomForestClassifier
            
        Returns:
            Self
        """
        self.feature_names = X.columns.tolist()
        self.classes_ = np.unique(y)
        
        # Create and fit the model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            **kwargs
        )
        
        self.model.fit(X, y)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of each class
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probabilities for each class
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit first.")
        
        return self.model.predict_proba(X)

class GradientBoostingModel(ProbabilityModel):
    """
    Gradient Boosting model for probability prediction
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                max_depth: int = 3, min_samples_split: int = 2,
                subsample: float = 1.0, random_state: int = 42,
                name: str = 'Gradient Boosting'):
        super().__init__(name)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'GradientBoostingModel':
        self.feature_names = X.columns.tolist()
        self.classes_ = np.unique(y)
        
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            
class ModelPipeline:
    """
    End-to-end pipeline for probability prediction
    """
    
    def __init__(self, name: str = 'Probability Pipeline'):
        self.name = name
        self.preprocessor = None
        self.feature_selector = None
        self.model = None
        self.threshold = 0.5
        self.feature_names = None
        self.target_name = None
        self.classes_ = None
    
    def fit(self, df: pd.DataFrame, target: str, 
           preprocessing_config: Dict[str, Any] = None,
           feature_selection: Dict[str, Any] = None,
           model: ProbabilityModel = None) -> 'ModelPipeline':
        """
        Fit the end-to-end pipeline
        
        Args:
            df: Input DataFrame
            target: Target column name
            preprocessing_config: Configuration for preprocessing
            feature_selection: Configuration for feature selection
            model: Probability prediction model
            
        Returns:
            Self
        """
        self.target_name = target
        
        # Split features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        self.classes_ = np.unique(y)
        
        # Preprocessing
        self.preprocessor = DataPreprocessor()
        X_processed = self.preprocessor.fit_transform(X, preprocessing_config)
        self.feature_names = X_processed.columns.tolist()
        
        # Feature selection (optional)
        if feature_selection:
            self.feature_selector = FeatureSelector()
            method = feature_selection.get('method', 'rfe')
            n_features = feature_selection.get('n_features', None)
            threshold = feature_selection.get('threshold', None)
            
            X_processed = self.feature_selector.select_features(
                X_processed, y, method=method, n_features=n_features, threshold=threshold
            )
            self.feature_names = X_processed.columns.tolist()
        
        # Model training
        if model is None:
            # Default to Random Forest if no model is provided
            self.model = RandomForestModel()
        else:
            self.model = model
        
        self.model.fit(X_processed, y)
        
        return self
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using the pipeline
        
        Args:
            df: Input DataFrame
            
        Returns:
            Array of probabilities for each class
        """
        if self.model is None:
            raise RuntimeError("Pipeline not fitted. Call fit first.")
        
        # Ensure target column is not in the input
        if self.target_name in df.columns:
            X = df.drop(columns=[self.target_name])
        else:
            X = df
        
        # Apply preprocessing
        X_processed = self.preprocessor.transform(X)
        
        # Apply feature selection (if used)
        if self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)
        
        # Predict probabilities
        return self.model.predict_proba(X_processed)
    
    def predict(self, df: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        Predict class labels using the pipeline
        
        Args:
            df: Input DataFrame
            threshold: Probability threshold for positive class
            
        Returns:
            Array of predicted class labels
        """
        if threshold is None:
            threshold = self.threshold
        
        probas = self.predict_proba(df)
        
        if probas.shape[1] == 2:  # Binary classification
            return (probas[:, 1] >= threshold).astype(int)
        else:  # Multi-class
            return np.argmax(probas, axis=1)
    
    def evaluate(self, df: pd.DataFrame, threshold: float = None) -> Dict[str, float]:
        """
        Evaluate pipeline on new data
        
        Args:
            df: Input DataFrame (must contain target column)
            threshold: Probability threshold for positive class
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.target_name not in df.columns:
            raise ValueError(f"Target column '{self.target_name}' not found in input DataFrame")
        
        if threshold is None:
            threshold = self.threshold
        
        # Split features and target
        X = df.drop(columns=[self.target_name])
        y = df[self.target_name]
        
        # Predict class labels
        y_prob = self.predict_proba(df)
        
        # For binary classification
        if y_prob.shape[1] == 2:
            y_prob_positive = y_prob[:, 1]
            y_pred = (y_prob_positive >= threshold).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y, y_prob_positive),
                'log_loss': log_loss(y, y_prob)
            }
            
        # For multi-class classification
        else:
            y_pred = np.argmax(y_prob, axis=1)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
                'log_loss': log_loss(y, y_prob)
            }
        
        return metrics
        
    def save(self, filepath: str) -> None:
        """
        Save pipeline to file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'ModelPipeline':
        """
        Load pipeline from file
        """
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"Pipeline loaded from {filepath}")
        return pipeline

