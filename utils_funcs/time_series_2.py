import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import os
import datetime
import warnings
from typing import List, Dict, Tuple, Union, Optional
import logging
from fbprophet import Prophet
import xgboost as XGBRegressor
import lightgbm as LGBMRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import pickle
import base64
from io import BytesIO
import jinja2
import pdfkit
import json
import sqlite3
from dataclasses import dataclass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# DATA LOADING AND PREPROCESSING MODULE
# =============================================================================

class DataLoader:
    """
    Class for loading time series data from various sources
    """
    
    @staticmethod
    def from_csv(filepath: str, date_col: str, value_cols: List[str], 
                 date_format: str = None, sep: str = ',') -> pd.DataFrame:
        """
        Load time series data from a CSV file
        
        Args:
            filepath: Path to the CSV file
            date_col: Name of the column containing dates
            value_cols: List of column names containing values to analyze
            date_format: Format of the date column
            sep: Separator in the CSV file
            
        Returns:
            DataFrame with DatetimeIndex
        """
        try:
            df = pd.read_csv(filepath, sep=sep)
            if date_format:
                df[date_col] = pd.to_datetime(df[date_col], format=date_format)
            else:
                df[date_col] = pd.to_datetime(df[date_col])
            
            df = df.set_index(date_col)
            return df[value_cols]
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    @staticmethod
    def from_excel(filepath: str, date_col: str, value_cols: List[str],
                  sheet_name: str = 0, date_format: str = None) -> pd.DataFrame:
        """
        Load time series data from an Excel file
        
        Args:
            filepath: Path to the Excel file
            date_col: Name of the column containing dates
            value_cols: List of column names containing values to analyze
            sheet_name: Name or index of the sheet to load
            date_format: Format of the date column
            
        Returns:
            DataFrame with DatetimeIndex
        """
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            if date_format:
                df[date_col] = pd.to_datetime(df[date_col], format=date_format)
            else:
                df[date_col] = pd.to_datetime(df[date_col])
            
            df = df.set_index(date_col)
            return df[value_cols]
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
    
    @staticmethod
    def from_database(connection_string: str, query: str, date_col: str, 
                      value_cols: List[str], date_format: str = None) -> pd.DataFrame:
        """
        Load time series data from a database using SQL query
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            date_col: Name of the column containing dates
            value_cols: List of column names containing values to analyze
            date_format: Format of the date column
            
        Returns:
            DataFrame with DatetimeIndex
        """
        try:
            conn = sqlite3.connect(connection_string)
            df = pd.read_sql_query(query, conn)
            
            if date_format:
                df[date_col] = pd.to_datetime(df[date_col], format=date_format)
            else:
                df[date_col] = pd.to_datetime(df[date_col])
            
            df = df.set_index(date_col)
            return df[value_cols]
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()


class TimeSeriesPreprocessor:
    """
    Class for preprocessing time series data
    """
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """
        Handle missing values in time series data
        
        Args:
            df: DataFrame with time series data
            method: Interpolation method ('linear', 'ffill', 'bfill', 'cubic', 'spline')
            
        Returns:
            DataFrame with missing values handled
        """
        if method == 'ffill':
            return df.fillna(method='ffill')
        elif method == 'bfill':
            return df.fillna(method='bfill')
        else:
            return df.interpolate(method=method)
    
    @staticmethod
    def resample(df: pd.DataFrame, freq: str = 'D', agg_func: str = 'mean') -> pd.DataFrame:
        """
        Resample time series data to a different frequency
        
        Args:
            df: DataFrame with time series data
            freq: Frequency for resampling ('D' for daily, 'W' for weekly, etc.)
            agg_func: Aggregation function for resampling ('mean', 'sum', 'min', 'max')
            
        Returns:
            Resampled DataFrame
        """
        if agg_func == 'mean':
            return df.resample(freq).mean()
        elif agg_func == 'sum':
            return df.resample(freq).sum()
        elif agg_func == 'min':
            return df.resample(freq).min()
        elif agg_func == 'max':
            return df.resample(freq).max()
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from time series data
        
        Args:
            df: DataFrame with time series data
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        result = df.copy()
        
        if method == 'iqr':
            for col in result.columns:
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                result.loc[result[col] < lower_bound, col] = np.nan
                result.loc[result[col] > upper_bound, col] = np.nan
        
        elif method == 'zscore':
            for col in result.columns:
                mean = result[col].mean()
                std = result[col].std()
                result.loc[abs(result[col] - mean) > threshold * std, col] = np.nan
        
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        # Interpolate missing values after removing outliers
        result = TimeSeriesPreprocessor.handle_missing_values(result)
        
        return result
    
    @staticmethod
    def decompose(df: pd.DataFrame, column: str, model: str = 'additive', period: int = None) -> pd.DataFrame:
        """
        Perform time series decomposition
        
        Args:
            df: DataFrame with time series data
            column: Column to decompose
            model: Decomposition model ('additive' or 'multiplicative')
            period: Period for decomposition (number of observations per period)
            
        Returns:
            DataFrame with decomposition components (trend, seasonal, residual)
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Auto-detect period if not provided
        if period is None:
            from scipy import signal
            data = df[column].dropna()
            freq, _ = signal.periodogram(data)
            if len(freq) > 1:
                period = int(1 / freq[1])
            else:
                # Default fallbacks based on index frequency
                if df.index.freqstr:
                    if 'D' in df.index.freqstr:
                        period = 7  # Weekly seasonality
                    elif 'H' in df.index.freqstr:
                        period = 24  # Daily seasonality
                    elif 'T' in df.index.freqstr or 'min' in df.index.freqstr.lower():
                        period = 60  # Hourly seasonality
                    elif 'M' in df.index.freqstr:
                        period = 12  # Yearly seasonality
                    else:
                        period = 12  # Default fallback
                else:
                    period = 12  # Default fallback
        
        decomposition = seasonal_decompose(df[column], model=model, period=period)
        
        result = pd.DataFrame({
            'observed': decomposition.observed,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        })
        
        return result


class TimeSeriesFeatureEngineering:
    """
    Class for engineering features specific to time series data
    """
    
    @staticmethod
    def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from datetime index
        
        Args:
            df: DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with additional time-based features
        """
        result = df.copy()
        result['hour'] = result.index.hour
        result['day_of_week'] = result.index.dayofweek
        result['day_of_month'] = result.index.day
        result['day_of_year'] = result.index.dayofyear
        result['week_of_year'] = result.index.isocalendar().week
        result['month'] = result.index.month
        result['quarter'] = result.index.quarter
        result['year'] = result.index.year
        result['is_weekend'] = result.index.dayofweek.isin([5, 6]).astype(int)
        return result
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Create lag features for time series
        
        Args:
            df: DataFrame with time series data
            columns: List of columns to create lag features for
            lags: List of lag periods
            
        Returns:
            DataFrame with additional lag features
        """
        result = df.copy()
        for col in columns:
            for lag in lags:
                result[f'{col}_lag_{lag}'] = result[col].shift(lag)
        return result
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, columns: List[str], 
                               windows: List[int], functions: List[str]) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: DataFrame with time series data
            columns: List of columns to create rolling features for
            windows: List of window sizes
            functions: List of aggregation functions ('mean', 'std', 'min', 'max')
            
        Returns:
            DataFrame with additional rolling window features
        """
        result = df.copy()
        for col in columns:
            for window in windows:
                for func in functions:
                    if func == 'mean':
                        result[f'{col}_rolling_{window}_mean'] = result[col].rolling(window=window).mean()
                    elif func == 'std':
                        result[f'{col}_rolling_{window}_std'] = result[col].rolling(window=window).std()
                    elif func == 'min':
                        result[f'{col}_rolling_{window}_min'] = result[col].rolling(window=window).min()
                    elif func == 'max':
                        result[f'{col}_rolling_{window}_max'] = result[col].rolling(window=window).max()
        return result
    
    @staticmethod
    def create_diff_features(df: pd.DataFrame, columns: List[str], periods: List[int]) -> pd.DataFrame:
        """
        Create differencing features
        
        Args:
            df: DataFrame with time series data
            columns: List of columns to create diff features for
            periods: List of differencing periods
            
        Returns:
            DataFrame with additional differencing features
        """
        result = df.copy()
        for col in columns:
            for period in periods:
                result[f'{col}_diff_{period}'] = result[col].diff(period)
        return result
    
    @staticmethod
    def create_cyclical_features(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
        """
        Create cyclical features using sine and cosine transformations
        
        Args:
            df: DataFrame with time series data
            col: Column name containing cyclical data (e.g., 'hour', 'day_of_week')
            period: Cycle period (e.g., 24 for hours, 7 for days of week)
            
        Returns:
            DataFrame with additional cyclical features
        """
        result = df.copy()
        result[f'{col}_sin'] = np.sin(2 * np.pi * result[col] / period)
        result[f'{col}_cos'] = np.cos(2 * np.pi * result[col] / period)
        return result


# =============================================================================
# TIME SERIES VISUALIZATION MODULE
# =============================================================================

class TimeSeriesVisualizer:
    """
    Class for creating various time series visualizations
    """
    
    @staticmethod
    def plot_time_series(df: pd.DataFrame, columns: List[str] = None, figsize: Tuple[int, int] = (15, 8),
                        title: str = 'Time Series Plot', ylabel: str = 'Value',
                        legend_loc: str = 'best', style: str = '-',
                        save_path: str = None) -> plt.Figure:
        """
        Plot time series data
        
        Args:
            df: DataFrame with time series data
            columns: List of columns to plot (if None, plot all columns)
            figsize: Figure size as (width, height)
            title: Plot title
            ylabel: Y-axis label
            legend_loc: Location of the legend
            style: Line style
            save_path: Path to save the figure (if None, figure is not saved)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if columns is None:
            columns = df.columns
        
        for col in columns:
            ax.plot(df.index, df[col], style, label=col)
        
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc=legend_loc)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_multiple_series(df: pd.DataFrame, columns: List[str] = None, 
                            figsize: Tuple[int, int] = (15, 10),
                            title: str = 'Multiple Time Series', 
                            save_path: str = None) -> plt.Figure:
        """
        Plot multiple time series as subplots
        
        Args:
            df: DataFrame with time series data
            columns: List of columns to plot (if None, plot all columns)
            figsize: Figure size as (width, height)
            title: Plot title
            save_path: Path to save the figure (if None, figure is not saved)
            
        Returns:
            Matplotlib figure object
        """
        if columns is None:
            columns = df.columns
        
        n_plots = len(columns)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        
        if n_plots == 1:
            axes = [axes]
        
        for i, col in enumerate(columns):
            axes[i].plot(df.index, df[col], '-')
            axes[i].set_title(col, fontsize=12)
            axes[i].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_decomposition(decomp_df: pd.DataFrame, figsize: Tuple[int, int] = (15, 12),
                          title: str = 'Time Series Decomposition',
                          save_path: str = None) -> plt.Figure:
        """
        Plot time series decomposition components
        
        Args:
            decomp_df: DataFrame with decomposition components
                      (must have 'observed', 'trend', 'seasonal', 'residual' columns)
            figsize: Figure size as (width, height)
            title: Plot title
            save_path: Path to save the figure (if None, figure is not saved)
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        components = ['observed', 'trend', 'seasonal', 'residual']
        
        for i, component in enumerate(components):
            axes[i].plot(decomp_df.index, decomp_df[component], '-')
            axes[i].set_title(component.capitalize(), fontsize=12)
            axes[i].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_acf_pacf(series: pd.Series, lags: int = 40, figsize: Tuple[int, int] = (15, 8),
                     title: str = 'ACF and PACF Plots', 
                     save_path: str = None) -> plt.Figure:
        """
        Plot ACF and PACF for a time series
        
        Args:
            series: Series with time series data
            lags: Number of lags to include
            figsize: Figure size as (width, height)
            title: Plot title
            save_path: Path to save the figure (if None, figure is not saved)
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title("Autocorrelation Function (ACF)", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05)
        axes[1].set_title("Partial Autocorrelation Function (PACF)", fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_seasonal_patterns(df: pd.DataFrame, column: str, freq: str = 'D',
                              figsize: Tuple[int, int] = (15, 10),
                              title: str = 'Seasonal Patterns',
                              save_path: str = None) -> plt.Figure:
        """
        Plot seasonal patterns in time series data
        
        Args:
            df: DataFrame with time series data
            column: Column to analyze
            freq: Frequency for grouping ('D' for day, 'M' for month, etc.)
            figsize: Figure size as (width, height)
            title: Plot title
            save_path: Path to save the figure (if None, figure is not saved)
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Daily/Weekly pattern
        if freq == 'D':
            df_daily = df.copy()
            df_daily['day_of_week'] = df_daily.index.dayofweek
            daily_pattern = df_daily.groupby('day_of_week')[column].mean()
            
            axes[0].bar(daily_pattern.index, daily_pattern.values)
            axes[0].set_title('Average by Day of Week', fontsize=12)
            axes[0].set_xticks(range(7))
            axes[0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            axes[0].grid(True, alpha=0.3)
        
        # Monthly pattern
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly.index.month
        monthly_pattern = df_monthly.groupby('month')[column].mean()
        
        axes[1].bar(monthly_pattern.index, monthly_pattern.values)
        axes[1].set_title('Average by Month', fontsize=12)
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        axes[1].grid(True, alpha=0.3)
        
        # Yearly pattern
        df_yearly = df.copy()
        df_yearly['year'] = df_yearly.index.year
        yearly_pattern = df_yearly.groupby('year')[column].mean()
        
        axes[2].plot(yearly_pattern.index, yearly_pattern.values, 'o-')
        axes[2].set_title('Average by Year', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_forecast(actual: pd.Series, forecast: pd.Series, 
                     pred_interval_lower: pd.Series = None, pred_interval_upper: pd.Series = None,
                     figsize: Tuple[int, int] = (15, 8),
                     title: str = 'Forecast vs Actual', 
                     save_path: str = None) -> plt.Figure:
        """
        Plot forecast vs actual values
        
        Args:
            actual: Series with actual values
            forecast: Series with forecasted values
            pred_interval_lower: Lower prediction interval (optional)
            pred_interval_upper: Upper prediction interval (optional)
            figsize: Figure size as (width, height)
            title: Plot title
            save_path: Path to save the figure (if None, figure is not saved)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual values
        ax.plot(actual.index, actual.values, 'b-', label='Actual')
        
        # Plot forecast
        ax.plot(forecast.index, forecast.values, 'r-', label='Forecast')
        
        # Plot prediction intervals if provided
        if pred_interval_lower is not None and pred_interval_upper is not None:
            ax.fill_between(forecast.index, pred_interval_lower.values, pred_interval_upper.values,
                           color='r', alpha=0.2, label='95% Confidence Interval')
        
        # Add a vertical line at the start of the forecast
        if (actual.index.max() < forecast.index.max()):
            first_forecast_date = forecast.index.min()
            ax.axvline(x=first_forecast_date, color='k', linestyle='--')
            ax.text(first_forecast_date, ax.get_ylim()[1], 'Forecast Start', 
                   horizontalalignment='center', verticalalignment='bottom')
        
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_interactive_series(df: pd.DataFrame, columns: List[str] = None,
                               title: str = 'Interactive Time Series Plot',
                               save_path: str = None) -> go.Figure:
        """
        Create an interactive time series plot using Plotly
        
        Args:
            df: DataFrame with time series data
            columns: List of columns to plot (if None, plot all columns)
            title: Plot title
            save_path: Path to save the figure (if None, figure is not saved)
            
        Returns:
            Plotly figure object
        """
        if columns is None:
            columns = df.columns
        
        fig = go.Figure()
        
        for col in columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            legend_title='Series',
            hovermode='x unified',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


# =============================================================================
# TIME SERIES MODELING MODULE
# =============================================================================

class BaseTimeSeriesModel:
    """
    Base class for time series models
    """
    
    def __init__(self, name: str = 'Base Model'):
        self.name = name
        self.model = None
        self.fitted = False
    
    def fit(self, train_data: pd.Series) -> None:
        """
        Fit the model to training data
        
        Args:
            train_data: Series with training data
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, n_periods: int, prediction_intervals: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Generate forecasts for future periods
        
        Args:
            n_periods: Number of periods to forecast
            prediction_intervals: Whether to include prediction intervals
            
        Returns:
            Forecast series or tuple of (forecast, lower_bound, upper_bound)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file
        
        Args:
            filepath: Path to save the model
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    @staticmethod
    def load_model(filepath: str) -> 'BaseTimeSeriesModel':
        """
        Load a model from a file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


class SARIMAXModel(BaseTimeSeriesModel):
    """
    SARIMAX model for time series forecasting
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1),
                seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                exog_variables: pd.DataFrame = None,
                name: str = 'SARIMAX Model'):
        """
        Initialize SARIMAX model
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
            exog_variables: Exogenous variables for training
            name: Model name
        """
        super().__init__(name)
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_variables = exog_variables
        self.exog_future = None
    
    def fit(self, train_data: pd.Series) -> None:
        """
        Fit SARIMAX model to training data
        
        Args:
            train_data: Series with training data
        """
        try:
            self.train_data = train_data
            
            self.model = SARIMAX(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                exog=self.exog_variables
            )
            
            self.result = self.model.fit(disp=False)
            self.fitted = True
            
            logger.info(f"SARIMAX model with order {self.order} and seasonal order "
                       f"{self.seasonal_order} fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting SARIMAX model: {e}")
            raise
    
    def set_future_exog(self, exog_future: pd.DataFrame) -> None:
        """
        Set exogenous variables for forecasting
        
        Args:
            exog_future: Exogenous variables for forecast horizon
        """
        self.exog_future = exog_future
    
    def predict(self, n_periods: int, prediction_intervals: bool = False, 
               alpha: float = 0.05) -> Union[pd.Series, Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Generate forecasts for future periods
        
        Args:
            n_periods: Number of periods to forecast
            prediction_intervals: Whether to include prediction intervals
            alpha: Significance level for prediction intervals
            
        Returns:
            Forecast series or tuple of (forecast, lower_bound, upper_bound)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        
        try:
            # Get the last date in the training data
            last_date = self.train_data.index[-1]
            
            # Generate future dates
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=n_periods,
                freq=pd.infer_freq(self.train_data.index)
            )
            
            # Generate forecast
            if prediction_intervals:
                forecast = self.result.get_forecast(steps=n_periods, exog=self.exog_future)
                forecast_mean = pd.Series(forecast.predicted_mean, index=future_dates)
                conf_int = forecast.conf_int(alpha=alpha)
                lower_bound = pd.Series(conf_int.iloc[:, 0].values, index=future_dates)
                upper_bound = pd.Series(conf_int.iloc[:, 1].values, index=future_dates)
                
                return forecast_mean, lower_bound, upper_bound
            else:
                forecast = self.result.get_forecast(steps=n_periods, exog=self.exog_future)
                forecast_mean = pd.Series(forecast.predicted_mean, index=future_dates)
                return forecast_mean
        
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise


class ExponentialSmoothingModel(BaseTimeSeriesModel):
    """
    Exponential Smoothing model for time series forecasting
    """
    
    def __init__(self, trend: str = None, seasonal: str = None, seasonal_periods: int = None,
                damped: bool = False, name: str = 'Exponential Smoothing Model'):
        """
        Initialize Exponential Smoothing model
        
        Args:
            trend: Type of trend component ('add' or 'mul' or None)
            seasonal: Type of seasonal component ('add' or 'mul' or None)
            seasonal_periods: Number of periods in a season
            damped: Whether to use damped trend
            name: Model name
        """
        super().__init__(name)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped = damped
    
    def fit(self, train_data: pd.Series) -> None:
        """
        Fit Exponential Smoothing model to training data
        
        Args:
            train_data: Series with training data
        """
        try:
            self.train_data = train_data
            
            self.model = ExponentialSmoothing(
                train_data,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped
            )
            
            self.result = self.model.fit()
            self.fitted = True
            
            logger.info(f"Exponential Smoothing model with trend={self.trend}, seasonal={self.seasonal}, "
                       f"seasonal_periods={self.seasonal_periods} fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting Exponential Smoothing model: {e}")
            raise
    
    def predict(self, n_periods: int, prediction_intervals: bool = False,
               alpha: float = 0.05) -> Union[pd.Series, Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Generate forecasts for future periods
        
        Args:
            n_periods: Number of periods to forecast
            prediction_intervals: Whether to include prediction intervals
            alpha: Significance level for prediction intervals
            
        Returns:
            Forecast series or tuple of (forecast, lower_bound, upper_bound)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        
        try:
            # Get the last date in the training data
            last_date = self.train_data.index[-1]
            
            # Generate future dates
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=n_periods,
                freq=pd.infer_freq(self.train_data.index)
            )
            
            # Generate forecast
            forecast_mean = pd.Series(
                self.result.forecast(n_periods),
                index=future_dates
            )
            
            if prediction_intervals:
                # For exponential smoothing, we'll use simulations to generate prediction intervals
                from scipy.stats import norm
                
                # Get residuals from the model
                residuals = self.result.resid
                residual_std = residuals.std()
                
                # Generate prediction intervals
                z_value = norm.ppf(1 - alpha / 2)
                lower_bound = forecast_mean - z_value * residual_std
                upper_bound = forecast_mean + z_value * residual_std
                
                return forecast_mean, lower_bound, upper_bound
            else:
                return forecast_mean
        
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise


class ProphetModel(BaseTimeSeriesModel):
    """
    Facebook Prophet model for time series forecasting
    """
    
    def __init__(self, seasonality_mode: str = 'additive', 
                yearly_seasonality: Union[bool, int] = 'auto',
                weekly_seasonality: Union[bool, int] = 'auto',
                daily_seasonality: Union[bool, int] = 'auto',
                holidays: pd.DataFrame = None,
                name: str = 'Prophet Model'):
        """
        Initialize Prophet model
        
        Args:
            seasonality_mode: Type of seasonality ('additive' or 'multiplicative')
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            holidays: DataFrame with holiday events
            name: Model name
        """
        super().__init__(name)
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.future_df = None
    
    def fit(self, train_data: pd.Series) -> None:
        """
        Fit Prophet model to training data
        
        Args:
            train_data: Series with training data
        """
        try:
            self.train_data = train_data
            
            # Prophet requires data in a specific format
            df = pd.DataFrame({
                'ds': train_data.index,
                'y': train_data.values
            })
            
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                holidays=self.holidays
            )
            
            self.result = self.model.fit(df)
            self.fitted = True
            
            logger.info(f"Prophet model with seasonality_mode={self.seasonality_mode} fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            raise
    
    def predict(self, n_periods: int, prediction_intervals: bool = False,
               frequency: str = 'D') -> Union[pd.Series, Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Generate forecasts for future periods
        
        Args:
            n_periods: Number of periods to forecast
            prediction_intervals: Whether to include prediction intervals
            frequency: Frequency of the forecast
            
        Returns:
            Forecast series or tuple of (forecast, lower_bound, upper_bound)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        
        try:
            # Create future dataframe
            if self.future_df is None:
                self.future_df = self.model.make_future_dataframe(
                    periods=n_periods,
                    freq=frequency
                )
            
            # Generate forecast
            forecast = self.model.predict(self.future_df)
            
            # Extract forecast dates
            future_dates = pd.DatetimeIndex(forecast['ds'].iloc[-n_periods:])
            
            # Create forecast series
            forecast_mean = pd.Series(
                forecast['yhat'].iloc[-n_periods:].values,
                index=future_dates
            )
            
            if prediction_intervals:
                lower_bound = pd.Series(
                    forecast['yhat_lower'].iloc[-n_periods:].values,
                    index=future_dates
                )
                
                upper_bound = pd.Series(
                    forecast['yhat_upper'].iloc[-n_periods:].values,
                    index=future_dates
                )
                
                return forecast_mean, lower_bound, upper_bound
            else:
                return forecast_mean
        
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise
    
    def get_components(self) -> pd.DataFrame:
        """
        Get the decomposed components of the forecast
        
        Returns:
            DataFrame with trend, seasonality components
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting components")
        
        forecast = self.model.predict(self.future_df)
        return forecast[['ds', 'trend', 'yearly', 'weekly', 'daily']]


class LSTMModel(BaseTimeSeriesModel):
    """
    LSTM model for time series forecasting
    """
    
    def __init__(self, n_steps: int = 3, n_features: int = 1, n_units: int = 50,
                dropout_rate: float = 0.2, epochs: int = 100, batch_size: int = 32,
                name: str = 'LSTM Model'):
        """
        Initialize LSTM model
        
        Args:
            n_steps: Number of time steps to consider as input
            n_features: Number of features
            n_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            epochs: Number of epochs for training
            batch_size: Batch size for training
            name: Model name
        """
        super().__init__(name)
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = None
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input
        
        Args:
            data: Array with time series data
            
        Returns:
            Tuple of (X, y) with input sequences and output values
        """
        X, y = [], []
        for i in range(len(data) - self.n_steps):
            X.append(data[i:(i + self.n_steps), 0])
            y.append(data[i + self.n_steps, 0])
        return np.array(X), np.array(y)
    
    def fit(self, train_data: pd.Series) -> None:
        """
        Fit LSTM model to training data
        
        Args:
            train_data: Series with training data
        """
        try:
            self.train_data = train_data
            
            # Scale data
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled = self.scaler.fit_transform(train_data.values.reshape(-1, 1))
            
            # Create sequences
            X_train, y_train = self._create_sequences(train_scaled)
            
            # Reshape input for LSTM [samples, time steps, features]
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], self.n_features)
            
            # Build model
            self.model = Sequential()
            self.model.add(LSTM(units=self.n_units, return_sequences=True, 
                              input_shape=(self.n_steps, self.n_features)))
            self.model.add(Dropout(self.dropout_rate))
            self.model.add(LSTM(units=self.n_units))
            self.model.add(Dropout(self.dropout_rate))
            self.model.add(Dense(1))
            
            self.model.compile(optimizer='adam', loss='mse')
            
            # Train model
            self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0
            )
            
            self.fitted = True
            
            logger.info(f"LSTM model with {self.n_units} units fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting LSTM model: {e}")
            raise
    
    def predict(self, n_periods: int, prediction_intervals: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Generate forecasts for future periods
        
        Args:
            n_periods: Number of periods to forecast
            prediction_intervals: Whether to include prediction intervals
            
        Returns:
            Forecast series or tuple of (forecast, lower_bound, upper_bound)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        
        try:
            # Get the last date in the training data
            last_date = self.train_data.index[-1]
            
            # Generate future dates
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=n_periods,
                freq=pd.infer_freq(self.train_data.index)
            )
            
            # Get the last n_steps values
            scaled_data = self.scaler.transform(self.train_data.values.reshape(-1, 1))
            input_data = scaled_data[-self.n_steps:].reshape(1, self.n_steps, self.n_features)
            
            # Generate forecast
            forecasts = []
            for _ in range(n_periods):
                # Predict next value
                next_pred = self.model.predict(input_data, verbose=0)[0, 0]
                forecasts.append(next_pred)
                
                # Update input data for next prediction
                input_data = np.append(input_data[:, 1:, :], 
                                      [[next_pred]], 
                                      axis=1)
            
            # Invert scaling
            forecasts = self.scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
            
            # Create forecast series
            forecast_mean = pd.Series(forecasts.flatten(), index=future_dates)
            
            if prediction_intervals:
                # For LSTM, we'll use a simple method for prediction intervals
                # based on training error
                
                # Get training error
                X_train, y_train = self._create_sequences(scaled_data)
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], self.n_features)
                y_pred = self.model.predict(X_train, verbose=0)
                
                # Invert scaling
                y_pred = self.scaler.inverse_transform(y_pred)
                y_train_inv = self.scaler.inverse_transform(y_train.reshape(-1, 1))
                
                # Calculate RMSE
                train_rmse = np.sqrt(mean_squared_error(y_train_inv, y_pred))
                
                # Create prediction intervals (assuming normal distribution of errors)
                lower_bound = forecast_mean - 1.96 * train_rmse
                upper_bound = forecast_mean + 1.96 * train_rmse
                
                return forecast_mean, lower_bound, upper_bound
            else:
                return forecast_mean
        
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise


class EnsembleModel(BaseTimeSeriesModel):
    """
    Ensemble model combining multiple time series models
    """
    
    def __init__(self, models: List[BaseTimeSeriesModel], weights: List[float] = None,
                name: str = 'Ensemble Model'):
        """
        Initialize Ensemble model
        
        Args:
            models: List of time series models
            weights: List of weights for each model (if None, equal weights are used)
            name: Model name
        """
        super().__init__(name)
        self.models = models
        
        if weights is None:
            # Equal weights
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def fit(self, train_data: pd.Series) -> None:
        """
        Fit each model in the ensemble to training data
        
        Args:
            train_data: Series with training data
        """
        try:
            self.train_data = train_data
            
            for i, model in enumerate(self.models):
                logger.info(f"Fitting model {i+1}/{len(self.models)}: {model.name}")
                model.fit(train_data)
            
            self.fitted = True
            
            logger.info(f"Ensemble model with {len(self.models)} models fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting Ensemble model: {e}")
            raise
    
    def predict(self, n_periods: int, prediction_intervals: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Generate forecasts by combining predictions from all models
        
        Args:
            n_periods: Number of periods to forecast
            prediction_intervals: Whether to include prediction intervals
            
        Returns:
            Forecast series or tuple of (forecast, lower_bound, upper_bound)
        """
        if not self.fitted:
            raise RuntimeError("Models must be fitted before forecasting")
        
        try:
            # Get forecasts from all models
            all_forecasts = []
            all_lower_bounds = []
            all_upper_bounds = []
            
            for i, (model, weight) in enumerate(zip(self.models, self.weights)):
                logger.info(f"Generating forecast from model {i+1}/{len(self.models)}: {model.name}")
                
                if prediction_intervals:
                    forecast, lower, upper = model.predict(n_periods, prediction_intervals=True)
                    all_forecasts.append(forecast * weight)
                    all_lower_bounds.append(lower * weight)
                    all_upper_bounds.append(upper * weight)
                else:
                    forecast = model.predict(n_periods)
                    all_forecasts.append(forecast * weight)
            
            # Combine forecasts
            combined_forecast = pd.Series(
                sum(forecast.values for forecast in all_forecasts),
                index=all_forecasts[0].index
            )
            
            if prediction_intervals:
                combined_lower = pd.Series(
                    sum(lower.values for lower in all_lower_bounds),
                    index=all_forecasts[0].index
                )
                
                combined_upper = pd.Series(
                    sum(upper.values for upper in all_upper_bounds),
                    index=all_forecasts[0].index
                )
                
                return combined_forecast, combined_lower, combined_upper
            else:
                return combined_forecast
        
        except Exception as e:
            logger.error(f"Error generating ensemble forecast: {e}")
            raise


class ModelEvaluator:
    """
    Class for evaluating time series forecasting models
    """
    
    @staticmethod
    def evaluate_model(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance using various metrics
        
        Args:
            actual: Series with actual values
            predicted: Series with predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Ensure indices match
        common_idx = actual.index.intersection(predicted.index)
        actual = actual.loc[common_idx]
        predicted = predicted.loc[common_idx]
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }
    
    @staticmethod
    def cross_validate(model: BaseTimeSeriesModel, data: pd.Series, n_splits: int = 5,
                      test_size: int = 30) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation
        
        Args:
            model: Model to evaluate
            data: Series with time series data
            n_splits: Number of cross-validation splits
            test_size: Size of the test set in each split
            
        Returns:
            Dictionary with evaluation metrics for each split
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        cv_metrics = {
            'MAE': [],
            'MSE': [],
            'RMSE': [],
            'R²': [],
            'MAPE': []
        }
        
        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Fit model on training data
            model.fit(train_data)
            
            # Predict for test period
            forecast = model.predict(len(test_data))
            
            # Evaluate
            metrics = ModelEvaluator.evaluate_model(test_data, forecast)
            
            for key, value in metrics.items():
                cv_metrics[key].append(value)
        
        return cv_metrics
    
    @staticmethod
    def compare_models(models: List[BaseTimeSeriesModel], train_data: pd.Series, 
                      test_data: pd.Series) -> pd.DataFrame:
        """
        Compare multiple models on the same test data
        
        Args:
            models: List of models to compare
            train_data: Series with training data
            test_data: Series with test data
            
        Returns:
            DataFrame with evaluation metrics for each model
        """
        results = []
        
        for model in models:
            # Fit model
            model.fit(train_data)
            
            # Generate forecast
            forecast = model.predict(len(test_data))
            
            # Evaluate
            metrics = ModelEvaluator.evaluate_model(test_data, forecast)
            metrics['Model'] = model.name
            
            results.append(metrics)
        
        return pd.DataFrame(results).set_index('Model')


# =============================================================================
# AUTOMATED REPORT GENERATION MODULE
# =============================================================================

class ReportGenerator:
    """
    Class for generating reports with forecasting results
    """
    
    def __init__(self, title: str = 'Time Series Forecasting Report',
                author: str = 'Automated System',
                company: str = 'Your Company',
                logo_path: str = None):
        """
        Initialize ReportGenerator
        
        Args:
            title: Report title
            author: Report author
            company: Company name
            logo_path: Path to company logo image
        """
        self.title = title
        self.author = author
        self.company = company
        self.logo_path = logo_path
        self.sections = []
        self.figures = []
        self.tables = []
    
    def add_section(self, title: str, content: str) -> None:
        """
        Add a text section to the report
        
        Args:
            title: Section title
            content: Section content (text)
        """
        self.sections.append({
            'type': 'text',
            'title': title,
            'content': content
        })
    
    def add_figure(self, fig: plt.Figure, title: str, description: str = None) -> None:
        """
        Add a figure to the report
        
        Args:
            fig: Matplotlib or Plotly figure
            title: Figure title
            description: Figure description
        """
        # For matplotlib figures
        if isinstance(fig, plt.Figure):
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            self.figures.append({
                'type': 'matplotlib',
                'title': title,
                'description': description,
                'image': img_str
            })
        
        # For plotly figures
        elif isinstance(fig, go.Figure):
            img_bytes = fig.to_image(format='png', width=1200, height=800)
            img_str = base64.b64encode(img_bytes).decode('utf-8')
            
            self.figures.append({
                'type': 'plotly',
                'title': title,
                'description': description,
                'image': img_str
            })
    
    def add_table(self, df: pd.DataFrame, title: str, description: str = None) -> None:
        """
        Add a table to the report
        
        Args:
            df: DataFrame with table data
            title: Table title
            description: Table description
        """
        self.tables.append({
            'title': title,
            'description': description,
            'html': df.to_html(classes='table table-striped', border=0)
        })
    
    def generate_html(self, filepath: str) -> None:
        """
        Generate HTML report
        
        Args:
            filepath: Path to save the HTML report
        """
        # Load template
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{{ title }}</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    color: #333;
                }
                .container {
                    width: 90%;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .header {
                    text-align: center;
                    margin-bottom: 40px;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 20px;
                }
                .company-logo {
                    max-width: 200px;
                    margin-bottom: 20px;
                }
                h1 {
                    font-size: 24px;
                    margin-bottom: 10px;
                }
                h2 {
                    font-size: 20px;
                    margin-top: 30px;
                    margin-bottom: 15px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }
                h3 {
                    font-size: 18px;
                    margin-top: 25px;
                    margin-bottom: 10px;
                }
                .section {
                    margin-bottom: 30px;
                }
                .figure {
                    margin: 20px 0;
                    text-align: center;
                }
                .figure img {
                    max-width: 100%;
                    height: auto;
                }
                .figure-caption {
                    font-style: italic;
                    color: #666;
                    margin-top: 10px;
                }
                .table-container {
                    margin: 20px 0;
                    overflow-x: auto;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th {
                    background-color: #f5f5f5;
                    font-weight: bold;
                    padding: 10px;
                    text-align: left;
                }
                td {
                    padding: 8px 10px;
                    border-top: 1px solid #ddd;
                }
                .table-caption {
                    font-style: italic;
                    color: #666;
                    margin-top: 10px;
                }
                .footer {
                    margin-top: 40px;
                    border-top: 1px solid #ddd;
                    padding-top: 20px;
                    text-align: center;
                    font-size: 14px;
                    color: #777;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    {% if logo_path %}
                    <img src="{{ logo_path }}" alt="{{ company }} Logo" class="company-logo">
                    {% endif %}
                    <h1>{{ title }}</h1>
                    <p>Generated by {{ author }} | {{ company }} | {{ date }}</p>
                </div>
                
                {% for section in sections %}
                <div class="section">
                    <h2>{{ section.title }}</h2>
                    <p>{{ section.content }}</p>
                </div>
                {% endfor %}
                
                {% for figure in figures %}
                <div class="section">
                    <h3>{{ figure.title }}</h3>
                    <div class="figure">
                        <img src="data:image/png;base64,{{ figure.image }}" alt="{{ figure.title }}">
                        {% if figure.description %}
                        <div class="figure-caption">{{ figure.description }}</div>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
                
                {% for table in tables %}
                <div class="section">
                    <h3>{{ table.title }}</h3>
                    <div class="table-container">
                        {{ table.html|safe }}
                        {% if table.description %}
                        <div class="table-caption">{{ table.description }}</div>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
                
                <div class="footer">
                    <p>Report generated on {{ date }}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Render template
        template = jinja2.Template(template_str)
        
        html = template.render(
            title=self.title,
            author=self.author,
            company=self.company,
            logo_path=self.logo_path,
            date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            sections=self.sections,
            figures=self.figures,
            tables=self.tables
        )
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"HTML report saved to {filepath}")
    
    def generate_pdf(self, filepath: str, from_html: str = None) -> None:
        """
        Generate PDF report
        
        Args:
            filepath: Path to save the PDF report
            from_html: Path to HTML file to convert (if None, generates HTML first)
        """
        try:
            # If HTML not provided, generate it first
            if from_html is None:
                html_path = filepath.replace('.pdf', '.html')
                self.generate_html(html_path)
                from_html = html_path
            
            # Convert HTML to PDF
            pdfkit.from_file(from_html, filepath)
            
            logger.info(f"PDF report saved to {filepath}")
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            print(f"Failed to generate PDF. Make sure wkhtmltopdf is installed. Error: {e}")
            

# =============================================================================
# UTILITY MODULE
# =============================================================================

class TimeSeriesUtils:
    """
    Utility functions for time series analysis
    """
    
    @staticmethod
    def check_stationarity(series: pd.Series, window: int = 10, 
                          figsize: Tuple[int, int] = (12, 8)) -> Tuple[Dict[str, float], plt.Figure]:
        """
        Check stationarity of time series using ADF test and rolling statistics
        
        Args:
            series: Series with time series data
            window: Window size for rolling statistics
            figsize: Figure size as (width, height)
            
        Returns:
            Tuple of (ADF test results, matplotlib figure)
        """
        # Rolling statistics
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        # ADF test
        adf_result = adfuller(series.dropna())
        
        adf_output = {
            'Test Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Lags Used': adf_result[2],
            'Number of Observations': adf_result[3],
            'Critical Values': adf_result[4]
        }
        
        is_stationary = adf_result[1] < 0.05
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(series, label='Original')
        ax.plot(rolling_mean, label=f'Rolling Mean (window={window})')
        ax.plot(rolling_std, label=f'Rolling Std (window={window})')
        
        ax.set_title(f'Rolling Statistics & ADF Test (p-value: {adf_result[1]:.4f})', fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        
        # Add stationarity result
        if is_stationary:
            ax.text(0.05, 0.95, 'Series is stationary (ADF p-value < 0.05)', 
                   transform=ax.transAxes, fontsize=12, 
                   bbox=dict(facecolor='green', alpha=0.1))
        else:
            ax.text(0.05, 0.95, 'Series is not stationary (ADF p-value >= 0.05)', 
                   transform=ax.transAxes, fontsize=12, 
                   bbox=dict(facecolor='red', alpha=0.1))
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        return adf_output, fig
    
    @staticmethod
    def find_optimal_parameters(series: pd.Series, max_p: int = 5, max_d: int = 2, 
                               max_q: int = 5, seasonal: bool = True,
                               m: int = 12) -> Dict[str, List[int]]:
        """
        Find optimal ARIMA/SARIMA parameters using AIC
        
        Args:
            series: Series with time series data
            max_p: Maximum p value to consider
            max_d: Maximum d value to consider
            max_q: Maximum q value to consider
            seasonal: Whether to include seasonal component
            m: Seasonality period
            
        Returns:
            Dictionary with optimal parameters
        """
        logger.info("Searching for optimal parameters... This may take a while.")
        
        # First find optimal d using ADF test
        adf_test, _ = TimeSeriesUtils.check_stationarity(series)
        
        # If already stationary, d=0, otherwise d=1 as default
        if adf_test['p-value'] < 0.05:
            d_values = [0]
        else:
            d_values = [1]
        
        best_aic = float('inf')
        best_params = None
        
        # Grid search for ARIMA parameters
        for p in range(max_p + 1):
            for d in d_values:
                for q in range(max_q + 1):
                    # Skip if both p and q are 0
                    if p == 0 and q == 0:
                        continue
                    
                    if seasonal:
                        for P in range(3):
                            for D in range(2):
                                for Q in range(3):
                                    # Skip if both P and Q are 0
                                    if P == 0 and Q == 0:
                                        continue
                                    
                                    try:
                                        model = SARIMAX(
                                            series,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, m)
                                        )
                                        result = model.fit(disp=False)
                                        current_aic = result.aic
                                        
                                        if current_aic < best_aic:
                                            best_aic = current_aic
                                            best_params = {
                                                'order': [p, d, q],
                                                'seasonal_order': [P, D, Q, m]
                                            }
                                            
                                            logger.info(f"New best SARIMA({p},{d},{q})({P},{D},{Q},{m}) with AIC: {current_aic:.2f}")
                                    
                                    except Exception as e:
                                        continue
                    else:
                        try:
                            model = SARIMAX(
                                series,
                                order=(p, d, q)
                            )
                            result = model.fit(disp=False)
                            current_aic = result.aic
                            
                            if current_aic < best_aic:
                                best_aic = current_aic
                                best_params = {
                                    'order': [p, d, q],
                                    'seasonal_order': [0, 0, 0, 0]
                                }
                                
                                logger.info(f"New best ARIMA({p},{d},{q}) with AIC: {current_aic:.2f}")
                        
                        except Exception as e:
                            continue
        
        if best_params is None:
            logger.warning("Could not find optimal parameters. Using default ARIMA(1,1,1).")
            best_params = {
                'order': [1, 1, 1],
                'seasonal_order': [0, 0, 0, 0]
            }
        
        return best_params


# Example usage module
def get_example_usage() -> str:
    """
    Return a string with example usage of the time series forecasting framework
    
    Returns:
        String with example code
    """
    return """
# Example usage of the Time Series Forecasting Framework

# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt
from time_series_framework import (
    DataLoader, TimeSeriesPreprocessor, TimeSeriesFeatureEngineering,
    TimeSeriesVisualizer, SARIMAXModel, ExponentialSmoothingModel, ProphetModel,
    ModelEvaluator, ReportGenerator, TimeSeriesUtils
)

# 1. Load and preprocess data
# ---------------------------
# Load data from CSV
data = DataLoader.from_csv(
    'sales_data.csv',
    date_col='date',
    value_cols=['sales'],
    date_format='%Y-%m-%d'
)

# Handle missing values and resample
data = TimeSeriesPreprocessor.handle_missing_values(data)
data = TimeSeriesPreprocessor.resample(data, freq='D', agg_func='mean')

# 2. Explore and visualize data
# -----------------------------
# Create time series plot
fig1 = TimeSeriesVisualizer.plot_time_series(
    data, 
    title='Sales Data',
    ylabel='Sales ($)'
)

# Decompose time series
decomp = TimeSeriesPreprocessor.decompose(data, 'sales', period=7)
fig2 = TimeSeriesVisualizer.plot_decomposition(decomp)

# Check stationarity
adf_results, fig3 = TimeSeriesUtils.check_stationarity(data['sales'])

# 3. Feature engineering
# ----------------------
# Create time features
data_features = TimeSeriesFeatureEngineering.create_time_features(data)

# Create lag features
data_features = TimeSeriesFeatureEngineering.create_lag_features(
    data_features, 
    columns=['sales'], 
    lags=[7, 14, 28]
)

# 4. Model training and forecasting
# ---------------------------------
# Split data into train and test
train_size = int(len(data) * 0.8)
train_data = data['sales'][:train_size]
test_data = data['sales'][train_size:]

# Find optimal SARIMA parameters
params = TimeSeriesUtils.find_optimal_parameters(
    train_data, 
    seasonal=True, 
    m=7
)

# Train SARIMA model
sarima_model = SARIMAXModel(
    order=tuple(params['order']), 
    seasonal_order=tuple(params['seasonal_order'])
)
sarima_model.fit(train_data)

# Train Exponential Smoothing model
exp_model = ExponentialSmoothingModel(
    trend='add', 
    seasonal='add', 
    seasonal_periods=7
)
exp_model.fit(train_data)

# Generate forecasts
sarima_forecast, lower, upper = sarima_model.predict(
    len(test_data), 
    prediction_intervals=True
)

# 5. Model evaluation
# ------------------
# Evaluate models
sarima_metrics = ModelEvaluator.evaluate_model(test_data, sarima_forecast)
print("SARIMA Model Metrics:", sarima_metrics)

# Compare models
models = [sarima_model, exp_model]
comparison = ModelEvaluator.compare_models(models, train_data, test_data)

# Plot forecasts
fig4 = TimeSeriesVisualizer.plot_forecast(
    test_data, 
    sarima_forecast, 
    lower, 
    upper,
    title='SARIMA Forecast vs Actual'
)

# 6. Generate report
# -----------------
report = ReportGenerator(
    title='Sales Forecasting Report',
    author='Data Science Team',
    company='Example Corp'
)

# Add sections
report.add_section(
    'Executive Summary',
    'This report presents the results of time series forecasting for daily sales data.'
)

report.add_section(
    'Data Overview',
    f'The dataset contains {len(data)} daily observations from {data.index.min()} to {data.index.max()}.'
)

# Add figures
report.add_figure(fig1, 'Sales Time Series', 'Daily sales data')
report.add_figure(fig2, 'Time Series Decomposition', 'Trend, seasonal, and residual components')
report.add_figure(fig4, 'Forecast Results', 'SARIMA model forecast with 95% prediction intervals')

# Add tables
report.add_table(comparison, 'Model Comparison', 'Performance metrics for different forecasting models')

# Generate HTML report
report.generate_html('sales_forecast_report.html')

# Generate PDF report
report.generate_pdf('sales_forecast_report.pdf')

print("Forecasting analysis completed and report generated.")
"""
