"""
Time Series Analysis Module
==========================
This module provides enhanced classes and functions for handling, analyzing, 
and preparing time series data for forecasting tasks.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple

# Optional imports moved inside functions to reduce initial load
# and make dependencies clearer for specific functionalities.

# --- Constants ---
DEFAULT_VALUE_COL = 'value'
DEFAULT_DATETIME_COL = 'datetime'

# --- Helper Functions ---

def load_ts_from_csv(
    filepath: str, 
    time_col: str, 
    value_col: str, 
    freq: Optional[str] = None, 
    datetime_format: Optional[str] = None,
    **kwargs: Any
) -> 'TimeSeries':
    """
    Loads time series data from a CSV file into a TimeSeries object.

    Parameters:
    ----------
    filepath : str
        Path to the CSV file.
    time_col : str
        Name of the column containing datetime information.
    value_col : str
        Name of the column containing the time series values.
    freq : str, optional
        Frequency code (e.g., 'D', 'H'). If None, pandas will attempt to infer it.
    datetime_format : str, optional
        The strftime format to parse the time column if it's not automatically recognized.
    **kwargs : Any
        Additional keyword arguments passed to pandas.read_csv.

    Returns:
    -------
    TimeSeries
        A TimeSeries object containing the loaded data.
        
    Raises:
    ------
    ValueError
        If the specified time or value columns are not found in the CSV.
    TypeError
        If the time column cannot be converted to datetime objects.
    """
    try:
        df = pd.read_csv(filepath, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {filepath}")

    if time_col not in df.columns:
        raise ValueError(f"Error: Time column '{time_col}' not found in CSV.")
    if value_col not in df.columns:
        raise ValueError(f"Error: Value column '{value_col}' not found in CSV.")

    try:
        # Attempt to convert to datetime, handling potential format issues
        if datetime_format:
            df[time_col] = pd.to_datetime(df[time_col], format=datetime_format)
        else:
            df[time_col] = pd.to_datetime(df[time_col]) 
    except Exception as e:
        raise TypeError(f"Error converting column '{time_col}' to datetime: {e}. "
                        "Consider specifying the 'datetime_format' parameter.")

    df = df.set_index(time_col)
    df = df[[value_col]].rename(columns={value_col: DEFAULT_VALUE_COL}) # Standardize value column name
    df = df.sort_index() # Ensure chronological order

    # Infer frequency if not provided
    if freq is None and pd.api.types.is_datetime64_any_dtype(df.index):
         inferred_freq = pd.infer_freq(df.index)
         if inferred_freq:
             freq = inferred_freq
             print(f"Inferred frequency: {freq}")
         else:
             print("Warning: Could not infer frequency. Operations requiring frequency might fail.")
             # Optionally, calculate median difference if needed, but freq remains None
             # time_diffs = df.index.to_series().diff().median()
             # print(f"Median time difference: {time_diffs}")


    # Ensure the index has a frequency set if possible
    if freq:
        df = df.asfreq(freq) # This fills missing timestamps with NaN

    return TimeSeries(df, freq=freq, value_col=DEFAULT_VALUE_COL)


# --- Core TimeSeries Class ---

class TimeSeries:
    """
    A class used to represent and manipulate time series data.
    
    Attributes:
    -----------
    data : pandas.DataFrame
        The time series data. Must have a DatetimeIndex.
    value_col : str
        The name of the column containing the primary time series values.
    freq : str or None
        The frequency of the time series (e.g., 'D', 'H'). None if irregular or unknown.
        
    Methods:
    --------
    __init__(data, value_col='value', freq=None)
        Initializes the TimeSeries object.
    plot(title='Time Series Plot', xlabel='Date', ylabel='Value', **kwargs)
        Plots the time series data.
    resample(new_freq, aggregation='mean')
        Resamples the time series to a new frequency.
    handle_missing(method='ffill', **kwargs)
        Handles missing values in the time series data.
    add_lag_features(lags: Union[int, List[int]])
        Adds lag features to the data.
    add_rolling_features(window: int, funcs: List[str] = ['mean', 'std'])
        Adds rolling window features (mean, std, min, max, etc.).
    add_datetime_features(features: List[str] = ['year', 'month', 'day', 'dayofweek', 'hour'])
        Adds features derived from the datetime index.
    summary()
        Prints summary statistics of the time series.
    get_value_column() -> pd.Series
        Returns the primary value column as a pandas Series.
    copy() -> 'TimeSeries'
        Creates a deep copy of the TimeSeries object.
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 value_col: str = DEFAULT_VALUE_COL, 
                 freq: Optional[str] = None):
        """
        Initializes the TimeSeries object.
        
        Parameters:
        ----------
        data : pandas.DataFrame
            DataFrame with a DatetimeIndex.
        value_col : str, optional
            Name of the column containing the primary time series values (default: 'value').
        freq : str, optional
            Frequency code (e.g., 'D', 'H'). If None, it attempts to infer from the index.
        
        Raises:
        ------
        TypeError
            If the index of the data is not a DatetimeIndex.
        ValueError
            If the specified value_col is not in the DataFrame columns.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Data index must be a pandas DatetimeIndex.")
        if value_col not in data.columns:
            raise ValueError(f"Value column '{value_col}' not found in data.")
            
        self.data = data.copy() # Work on a copy
        self.value_col = value_col
        
        # Set or infer frequency
        if freq:
            self.freq = freq
            # Ensure index frequency matches if possible
            if self.data.index.freq is None or self.data.index.freq != freq:
                 try:
                     self.data = self.data.asfreq(self.freq)
                 except ValueError as e:
                     print(f"Warning: Could not set frequency '{self.freq}' on index. "
                           f"Index might be irregular or have duplicates. Error: {e}")
                     self.freq = self.data.index.freq # Use original index freq if asfreq fails
        else:
            self.freq = self.data.index.freq or pd.infer_freq(self.data.index)
            if self.freq:
                 print(f"Inferred frequency: {self.freq}")
            else:
                 print("Warning: Could not infer frequency for the time series.")

    def get_value_column(self) -> pd.Series:
        """Returns the primary value column as a pandas Series."""
        return self.data[self.value_col]

    def plot(self, title: str = 'Time Series Plot', xlabel: str = 'Date', ylabel: Optional[str] = None, **kwargs: Any):
        """
        Plots the time series data using matplotlib.
        
        Parameters:
        ----------
        title : str, optional
            Title of the plot.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis. Defaults to the value column name.
        **kwargs : Any
            Additional keyword arguments passed to pandas.Series.plot().
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed. Plotting unavailable. "
                  "Install with: pip install matplotlib")
            return
            
        plt.figure(figsize=kwargs.pop('figsize', (12, 6))) # Default size if not provided
        self.get_value_column().plot(**kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel if ylabel else self.value_col)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
    def resample(self, new_freq: str, aggregation: Union[str, callable] = 'mean') -> 'TimeSeries':
        """
        Resamples the time series to a new frequency.
        
        Parameters:
        ----------
        new_freq : str
            New frequency code (e.g., 'W', 'M', 'Q').
        aggregation : str or callable, optional
            Method for aggregating data ('mean', 'sum', 'median', 'min', 'max', or a custom function). 
            Default is 'mean'.
            
        Returns:
        -------
        TimeSeries
            New TimeSeries object with resampled data. Only includes the value column.
            
        Raises:
        ------
        ValueError
            If resampling fails (e.g., due to incompatible frequency).
        """
        try:
            resampled_data = self.data[[self.value_col]].resample(new_freq).agg(aggregation)
            # Retain original value column name if it was default, otherwise keep it as 'value'
            col_name = self.value_col if self.value_col == DEFAULT_VALUE_COL else DEFAULT_VALUE_COL
            if col_name != DEFAULT_VALUE_COL:
                 resampled_data = resampled_data.rename(columns={self.value_col: col_name})
            else: # Aggregation might rename single column, ensure it's 'value'
                 resampled_data.columns = [DEFAULT_VALUE_COL]

            return TimeSeries(resampled_data, freq=new_freq, value_col=resampled_data.columns[0])
        except ValueError as e:
            raise ValueError(f"Error resampling to frequency '{new_freq}': {e}")

    def handle_missing(self, method: str = 'ffill', **kwargs: Any) -> 'TimeSeries':
        """
        Handles missing values (NaN) in the time series data (modifies inplace).

        Parameters:
        ----------
        method : str, optional
            Method to fill missing values:
            - 'ffill': Forward fill
            - 'bfill': Backward fill
            - 'interpolate': Linear interpolation (default)
            - 'mean': Fill with the mean of the series
            - 'median': Fill with the median of the series
            - 'zero': Fill with 0
            - 'drop': Drop rows with missing values
            Default is 'ffill'.
        **kwargs : Any
            Additional keyword arguments passed to the underlying pandas method 
            (e.g., `limit` for ffill/bfill, `order` for interpolate).

        Returns:
        -------
        TimeSeries
            The modified TimeSeries object (self) for chaining.
            
        Raises:
        ------
        ValueError
            If an invalid method is provided.
        """
        value_series = self.get_value_column()
        initial_nan_count = value_series.isna().sum()
        if initial_nan_count == 0:
            print("No missing values found.")
            return self

        if method == 'ffill':
            self.data[self.value_col] = value_series.ffill(**kwargs)
        elif method == 'bfill':
            self.data[self.value_col] = value_series.bfill(**kwargs)
        elif method == 'interpolate':
            self.data[self.value_col] = value_series.interpolate(method='linear', **kwargs) # Default linear
        elif method == 'mean':
            fill_value = value_series.mean()
            self.data[self.value_col] = value_series.fillna(fill_value)
        elif method == 'median':
            fill_value = value_series.median()
            self.data[self.value_col] = value_series.fillna(fill_value)
        elif method == 'zero':
            self.data[self.value_col] = value_series.fillna(0)
        elif method == 'drop':
            self.data = self.data.dropna(subset=[self.value_col])
            print(f"Dropped {initial_nan_count} rows with missing values.")
            # Frequency might become invalid after dropping rows
            self.freq = pd.infer_freq(self.data.index)
            if not self.freq: print("Warning: Frequency might be lost after dropping NaNs.")
            return self
        else:
            raise ValueError(f"Invalid missing value handling method: '{method}'. "
                             "Choose from 'ffill', 'bfill', 'interpolate', 'mean', 'median', 'zero', 'drop'.")

        filled_count = initial_nan_count - self.get_value_column().isna().sum()
        print(f"Filled {filled_count} missing value(s) using method '{method}'.")
        return self # Return self for chaining

    def add_lag_features(self, lags: Union[int, List[int]]) -> 'TimeSeries':
        """
        Adds lag features for the value column to the DataFrame (modifies inplace).

        Parameters:
        ----------
        lags : int or List[int]
            The lag(s) to create. E.g., 1 for lag-1, [1, 3, 5] for lag-1, lag-3, lag-5.

        Returns:
        -------
        TimeSeries
            The modified TimeSeries object (self) for chaining.
        """
        if isinstance(lags, int):
            lags = [lags]
        
        value_series = self.get_value_column()
        for lag in lags:
            if lag <= 0:
                print(f"Warning: Skipping non-positive lag value: {lag}")
                continue
            col_name = f"{self.value_col}_lag_{lag}"
            self.data[col_name] = value_series.shift(lag)
            print(f"Added feature: {col_name}")
            
        return self

    def add_rolling_features(self, window: int, funcs: List[str] = ['mean', 'std']) -> 'TimeSeries':
        """
        Adds rolling window features for the value column (modifies inplace).

        Parameters:
        ----------
        window : int
            The size of the rolling window.
        funcs : List[str], optional
            List of aggregation functions to apply ('mean', 'std', 'min', 'max', 'median', 'sum').
            Default is ['mean', 'std'].

        Returns:
        -------
        TimeSeries
            The modified TimeSeries object (self) for chaining.
            
        Raises:
        ------
        ValueError
            If window size is not positive.
        """
        if window <= 0:
            raise ValueError("Window size must be a positive integer.")
            
        value_series = self.get_value_column()
        valid_funcs = {'mean', 'std', 'min', 'max', 'median', 'sum'} # Add more pandas rolling funcs if needed
        
        for func_name in funcs:
            if func_name not in valid_funcs:
                print(f"Warning: Skipping invalid rolling function '{func_name}'. Valid options: {valid_funcs}")
                continue
                
            col_name = f"{self.value_col}_roll_{window}_{func_name}"
            try:
                # Get the rolling object
                rolling_obj = value_series.rolling(window=window)
                # Apply the function by name
                self.data[col_name] = getattr(rolling_obj, func_name)()
                print(f"Added feature: {col_name}")
            except AttributeError:
                print(f"Warning: Could not apply rolling function '{func_name}'.")
            except Exception as e:
                 print(f"Error adding rolling feature {col_name}: {e}")

        return self

    def add_datetime_features(self, features: List[str] = ['year', 'month', 'day', 'dayofweek', 'hour', 'weekofyear', 'quarter']) -> 'TimeSeries':
        """
        Adds features derived from the datetime index (modifies inplace).

        Parameters:
        ----------
        features : List[str], optional
            List of datetime attributes to extract. Common options:
            'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 
            'weekday', 'dayofweek', 'dayofyear', 'weekofyear', 'week', 'quarter'.
            Default: ['year', 'month', 'day', 'dayofweek', 'hour', 'weekofyear', 'quarter']

        Returns:
        -------
        TimeSeries
            The modified TimeSeries object (self) for chaining.
        """
        idx = self.data.index
        allowed_features = ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 
                            'weekday', 'dayofweek', 'dayofyear', 'weekofyear', 'week', 'quarter']
                            
        for feature in features:
            if feature in allowed_features:
                try:
                    self.data[f"dt_{feature}"] = getattr(idx, feature)
                    print(f"Added feature: dt_{feature}")
                except AttributeError:
                    print(f"Warning: Could not extract datetime feature '{feature}'. It might not be available for the index.")
            else:
                 print(f"Warning: Skipping unsupported datetime feature '{feature}'. Allowed: {allowed_features}")
                 
        return self

    def summary(self):
        """Prints summary statistics of the time series value column."""
        print("--- Time Series Summary ---")
        print(f"Value Column: {self.value_col}")
        print(f"Frequency: {self.freq}")
        print(f"Time Range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Number of Observations: {len(self.data)}")
        print(f"Missing Values: {self.get_value_column().isna().sum()}")
        print("\nValue Column Statistics:")
        print(self.get_value_column().describe())
        print("--------------------------")

    def copy(self) -> 'TimeSeries':
        """Creates a deep copy of the TimeSeries object."""
        return TimeSeries(self.data.copy(), value_col=self.value_col, freq=self.freq)

# --- Time Series Analysis Tools ---

class TimeSeriesAnalyzer:
    """
    A class providing static methods for analyzing TimeSeries objects.
    """
    
    @staticmethod
    def calculate_trend(time_series: TimeSeries) -> Tuple[float, float]:
        """
        Calculates the linear trend of the time series using Ordinary Least Squares (OLS).
        
        Parameters:
        ----------
        time_series : TimeSeries
            TimeSeries object containing the data.
            
        Returns:
        -------
        Tuple[float, float]
            A tuple containing the slope (trend) and the intercept of the linear regression line.
            Returns (nan, nan) if calculation fails (e.g., insufficient data).
            
        Requires:
        ---------
        statsmodels: Install with `pip install statsmodels`
        """
        try:
            import statsmodels.api as sm
        except ImportError:
            print("Warning: statsmodels not installed. Trend calculation unavailable. "
                  "Install with: pip install statsmodels")
            return (np.nan, np.nan)

        y = time_series.get_value_column().dropna() # Use only non-NaN values
        if len(y) < 2:
            print("Warning: Insufficient data points (< 2) to calculate trend.")
            return (np.nan, np.nan)
            
        # Use numerical representation of time for regression
        # Using seconds since epoch for consistency
        time_values = y.index.view('int64') // 10**9 
        
        X = sm.add_constant(time_values) # Add intercept term
        
        try:
            model = sm.OLS(y, X).fit()
            intercept, slope = model.params
            print(f"Linear Trend: Slope={slope:.4f}, Intercept={intercept:.4f}")
            return slope, intercept
        except Exception as e:
            print(f"Error calculating trend: {e}")
            return (np.nan, np.nan)

    @staticmethod
    def detect_anomalies(time_series: TimeSeries, method: str = 'zscore', threshold: float = 3.0, window: Optional[int] = None) -> pd.DataFrame:
        """
        Detects anomalies in the time series value column.
        
        Parameters:
        ----------
        time_series : TimeSeries
            TimeSeries object containing the data.
        method : str, optional
            Method for anomaly detection:
            - 'zscore': Uses Z-score based on the entire series mean/std.
            - 'rolling_zscore': Uses Z-score based on a rolling window mean/std. Requires 'window'.
            - 'iqr': Uses the Interquartile Range (IQR) method.
            Default is 'zscore'.
        threshold : float, optional
            Threshold for detection. 
            - For 'zscore'/'rolling_zscore': Number of standard deviations from the mean.
            - For 'iqr': Multiplier for the IQR (typically 1.5 or 3.0).
            Default is 3.0.
        window : int, optional
            The window size required for the 'rolling_zscore' method. Ignored otherwise.
            
        Returns:
        -------
        pandas.DataFrame
            DataFrame containing the detected anomalies (index, value). Empty if none found.
            
        Raises:
        ------
        ValueError
            If an invalid method is provided or if 'window' is missing for 'rolling_zscore'.
        """
        values = time_series.get_value_column().dropna()
        if values.empty:
            return pd.DataFrame() # Return empty DataFrame if no data

        anomalies = pd.Series(dtype=bool) # Initialize empty boolean Series

        if method == 'zscore':
            mean = values.mean()
            std = values.std()
            if std == 0: # Avoid division by zero
                 print("Warning: Standard deviation is zero. Cannot calculate Z-scores.")
                 return pd.DataFrame()
            z_scores = np.abs((values - mean) / std)
            anomalies = z_scores > threshold
            
        elif method == 'rolling_zscore':
            if window is None or window <= 0:
                raise ValueError("A positive 'window' size must be provided for 'rolling_zscore' method.")
            rolling_mean = values.rolling(window=window, center=True, min_periods=1).mean()
            rolling_std = values.rolling(window=window, center=True, min_periods=1).std()
            # Avoid division by zero in rolling std
            rolling_std = rolling_std.replace(0, np.nan).ffill().bfill() # Fill zero stds
            if rolling_std.isna().all():
                 print("Warning: Rolling standard deviation could not be calculated (all zero or NaN). Cannot detect anomalies.")
                 return pd.DataFrame()

            z_scores = np.abs((values - rolling_mean) / rolling_std)
            anomalies = z_scores > threshold
            
        elif method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            anomalies = (values < lower_bound) | (values > upper_bound)
            
        else:
            raise ValueError(f"Invalid anomaly detection method: '{method}'. "
                             "Choose from 'zscore', 'rolling_zscore', 'iqr'.")

        detected_anomalies = time_series.data.loc[anomalies, [time_series.value_col]]
        print(f"Detected {len(detected_anomalies)} anomalies using method '{method}' with threshold/window '{threshold}/{window if method=='rolling_zscore' else 'N/A'}'.")
        return detected_anomalies

    @staticmethod
    def test_stationarity(time_series: TimeSeries, significance_level: float = 0.05):
        """
        Performs the Augmented Dickey-Fuller (ADF) test for stationarity.
        
        Prints the test results, including the test statistic, p-value, and critical values.
        Indicates whether the series is likely stationary based on the p-value.

        Parameters:
        ----------
        time_series : TimeSeries
            TimeSeries object containing the data.
        significance_level : float, optional
            The significance level (alpha) to compare the p-value against (default is 0.05).

        Requires:
        ---------
        statsmodels: Install with `pip install statsmodels`
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            print("Warning: statsmodels not installed. Stationarity test unavailable. "
                  "Install with: pip install statsmodels")
            return

        values = time_series.get_value_column().dropna() # Test works best on non-missing data
        if len(values) < 4: # ADF needs a few data points
            print("Warning: Insufficient data points to perform ADF test.")
            return

        print("\n--- Augmented Dickey-Fuller Test ---")
        try:
            result = adfuller(values)
            print(f'ADF Statistic: {result[0]:.4f}')
            print(f'p-value: {result[1]:.4f}')
            print(f'# Lags Used: {result[2]}')
            print(f'# Observations Used: {result[3]}')
            print('Critical Values:')
            for key, value in result[4].items():
                print(f'\t{key}: {value:.4f}')
            
            # Interpretation
            if result[1] <= significance_level:
                print(f"\nResult: Reject the null hypothesis (H0) at {significance_level*100}% significance level.")
                print("The time series is likely stationary.")
            else:
                print(f"\nResult: Fail to reject the null hypothesis (H0) at {significance_level*100}% significance level.")
                print("The time series is likely non-stationary.")
        except Exception as e:
            print(f"Error performing ADF test: {e}")
        print("------------------------------------")


    @staticmethod
    def plot_decomposition(time_series: TimeSeries, model: str = 'additive', period: Optional[int] = None):
        """
        Performs and plots time series decomposition (Trend, Seasonality, Residuals).

        Parameters:
        ----------
        time_series : TimeSeries
            TimeSeries object containing the data.
        model : str, optional
            Type of decomposition model ('additive' or 'multiplicative'). Default is 'additive'.
        period : int, optional
            The period of the seasonality. If None, it attempts to infer from the frequency 
            (e.g., 7 for daily, 12 for monthly). Required if frequency is ambiguous or missing.

        Requires:
        ---------
        statsmodels: Install with `pip install statsmodels`
        matplotlib: Install with `pip install matplotlib`
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: statsmodels or matplotlib not installed. Decomposition unavailable. "
                  "Install with: pip install statsmodels matplotlib")
            return

        values = time_series.get_value_column().dropna()
        
        # Attempt to infer period if not provided
        if period is None:
            if time_series.freq:
                freq_str = time_series.freq.upper()
                if 'D' in freq_str: period = 7
                elif 'W' in freq_str: period = 52 # Approx
                elif 'M' in freq_str: period = 12
                elif 'Q' in freq_str: period = 4
                elif 'A' in freq_str or 'Y' in freq_str: period = 1 # Yearly has no sub-period typically
                elif 'H' in freq_str: period = 24
                elif 'T' in freq_str or 'MIN' in freq_str: period = 60
                elif 'S' in freq_str: period = 60
                
                if period:
                    print(f"Inferred seasonality period: {period} based on frequency '{time_series.freq}'")
                else:
                     print("Warning: Could not infer seasonality period from frequency. Decomposition might be inaccurate. Please specify 'period'.")
                     return
            else:
                 print("Warning: Time series frequency is unknown. Cannot infer seasonality period. Please specify 'period'.")
                 return

        if len(values) < 2 * period:
             print(f"Warning: Time series length ({len(values)}) is less than twice the period ({period}). Decomposition may be unreliable or fail.")
             # Optionally return here, or let seasonal_decompose handle the error
             # return

        try:
            decomposition = seasonal_decompose(values, model=model, period=period, extrapolate_trend='freq')
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            
            decomposition.observed.plot(ax=ax1, legend=False)
            ax1.set_ylabel('Observed')
            decomposition.trend.plot(ax=ax2, legend=False)
            ax2.set_ylabel('Trend')
            decomposition.seasonal.plot(ax=ax3, legend=False)
            ax3.set_ylabel('Seasonal')
            decomposition.resid.plot(ax=ax4, legend=False)
            ax4.set_ylabel('Residual')
            
            fig.suptitle(f'Time Series Decomposition ({model.capitalize()} Model, Period={period})', y=0.95)
            plt.xlabel('Date')
            plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout to prevent title overlap
            plt.show()
            
        except ValueError as e:
            print(f"Error during decomposition: {e}. Check if the series length is sufficient for the period.")
        except Exception as e:
            print(f"An unexpected error occurred during decomposition: {e}")


    @staticmethod
    def plot_acf_pacf(time_series: TimeSeries, lags: Optional[int] = None, alpha: float = 0.05):
        """
        Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

        Parameters:
        ----------
        time_series : TimeSeries
            TimeSeries object containing the data.
        lags : int, optional
            Number of lags to include in the plots. If None, it defaults to a value 
            based on the series length (min(40, N/2 - 1)).
        alpha : float, optional
            Significance level for confidence intervals (default is 0.05 for 95% CI).

        Requires:
        ---------
        statsmodels: Install with `pip install statsmodels`
        matplotlib: Install with `pip install matplotlib`
        """
        try:
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: statsmodels or matplotlib not installed. ACF/PACF plots unavailable. "
                  "Install with: pip install statsmodels matplotlib")
            return
            
        values = time_series.get_value_column().dropna()
        if values.empty:
            print("Warning: Cannot plot ACF/PACF for empty series.")
            return

        if lags is None:
            # Default lags: commonly used heuristic, up to 40 or half the length
            n_obs = len(values)
            lags = min(40, int(n_obs / 2) - 1)
            if lags < 1: lags = 1 # Ensure at least 1 lag if possible
            print(f"Using default number of lags: {lags}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        try:
            plot_acf(values, lags=lags, alpha=alpha, ax=axes[0])
            axes[0].set_title('Autocorrelation Function (ACF)')
            
            # PACF calculation might require more points than lags
            if len(values) > lags + 1 :
                 plot_pacf(values, lags=lags, alpha=alpha, ax=axes[1], method='ols') # 'ols' is common
                 axes[1].set_title('Partial Autocorrelation Function (PACF)')
            else:
                 axes[1].set_title('PACF (Not enough data for calculation)')
                 print(f"Warning: Not enough data points ({len(values)}) to calculate PACF for {lags} lags.")


            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting ACF/PACF: {e}")


# --- Evaluation Metrics ---

def calculate_forecast_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """
    Calculates common forecast evaluation metrics.

    Parameters:
    ----------
    actual : pd.Series
        The actual observed values.
    predicted : pd.Series
        The predicted values from the forecast model.

    Returns:
    -------
    Dict[str, float]
        A dictionary containing:
        - 'MAE': Mean Absolute Error
        - 'MSE': Mean Squared Error
        - 'RMSE': Root Mean Squared Error
        - 'MAPE': Mean Absolute Percentage Error (returns np.inf if actual contains 0)
        - 'SMAPE': Symmetric Mean Absolute Percentage Error 
        
    Raises:
    ------
    ValueError
        If actual and predicted series have different lengths or indices.
    """
    if not isinstance(actual, pd.Series): actual = pd.Series(actual)
    if not isinstance(predicted, pd.Series): predicted = pd.Series(predicted)

    # Align series based on index
    predicted = predicted.reindex(actual.index)

    # Drop NaNs that might result from alignment or original data
    combined = pd.DataFrame({'actual': actual, 'predicted': predicted}).dropna()
    
    if len(combined['actual']) == 0:
        print("Warning: No overlapping non-NaN data points between actual and predicted series.")
        return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'SMAPE': np.nan}

    y_true = combined['actual']
    y_pred = combined['predicted']

    metrics = {}
    metrics['MAE'] = np.mean(np.abs(y_true - y_pred))
    metrics['MSE'] = np.mean((y_true - y_pred)**2)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    
    # MAPE - Handle potential division by zero
    mask = y_true != 0
    if np.any(mask):
        metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics['MAPE'] = np.inf # Or np.nan, depending on preference
        
    # SMAPE - Symmetric version, less sensitive to zeros in actual
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Handle cases where both actual and predicted are zero
    diff = np.abs(y_true - y_pred)
    # Avoid division by zero where both actual and predicted are zero
    smape_vals = np.divide(diff, denominator, out=np.zeros_like(diff, dtype=float), where=denominator!=0)
    metrics['SMAPE'] = np.mean(smape_vals) * 100

    print("\n--- Forecast Evaluation Metrics ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("---------------------------------")
        
    return metrics


# --- Example Usage ---
if __name__ == "__main__":
    
    # Create sample data
    date_rng = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
    data_values = np.random.randn(len(date_rng)).cumsum() + 50 # Random walk
    # Add some seasonality and noise
    data_values += np.sin(np.arange(len(date_rng)) * 2 * np.pi / 7) * 5 
    sample_df = pd.DataFrame(data_values, index=date_rng, columns=['value'])
    
    # Introduce some missing values and an anomaly
    sample_df.iloc[10:15, 0] = np.nan
    sample_df.iloc[30, 0] = 100 # Anomaly
    
    print("--- Creating TimeSeries Object ---")
    ts = TimeSeries(sample_df, value_col='value', freq='D')
    ts.summary()

    print("\n--- Handling Missing Values (ffill) ---")
    ts.handle_missing(method='ffill')
    print(f"Missing values after ffill: {ts.get_value_column().isna().sum()}")

    print("\n--- Plotting ---")
    ts.plot(title="Sample Time Series with Anomaly")

    print("\n--- Feature Engineering ---")
    ts.add_lag_features([1, 7])
    ts.add_rolling_features(window=7, funcs=['mean', 'std'])
    ts.add_datetime_features(['dayofweek', 'month'])
    print("\nDataFrame with new features (first 5 rows):")
    print(ts.data.head())

    print("\n--- Analysis ---")
    analyzer = TimeSeriesAnalyzer()
    
    print("\n* Trend Calculation *")
    slope, intercept = analyzer.calculate_trend(ts)
    
    print("\n* Anomaly Detection (IQR) *")
    anomalies = analyzer.detect_anomalies(ts, method='iqr', threshold=1.5)
    print("Detected Anomalies:")
    print(anomalies)

    print("\n* Stationarity Test *")
    analyzer.test_stationarity(ts) # Original series likely non-stationary

    print("\n* Decomposition Plot *")
    # Create a copy without extra features for cleaner decomposition
    ts_for_decomp = TimeSeries(ts.data[[ts.value_col]], value_col=ts.value_col, freq=ts.freq)
    analyzer.plot_decomposition(ts_for_decomp, model='additive', period=7) 

    print("\n* ACF/PACF Plots *")
    analyzer.plot_acf_pacf(ts_for_decomp, lags=30)

    print("\n--- Resampling ---")
    ts_weekly = ts.resample('W', aggregation='mean')
    ts_weekly.summary()
    ts_weekly.plot(title="Weekly Resampled Time Series")

    print("\n--- Forecast Evaluation Example ---")
    # Dummy forecast (e.g., naive forecast shifted)
    actual_vals = ts.get_value_column().iloc[1:]
    predicted_vals = ts.get_value_column().shift(1).iloc[1:]
    metrics = calculate_forecast_metrics(actual_vals, predicted_vals)
    
    # Example loading from CSV (requires a dummy file)
    # try:
    #     # Create a dummy CSV for testing
    #     dummy_csv_path = "dummy_ts.csv"
    #     ts.data[[ts.value_col]].reset_index().rename(columns={'index': 'timestamp'}).to_csv(dummy_csv_path, index=False)
    #     print(f"\n--- Loading from CSV ({dummy_csv_path}) ---")
    #     loaded_ts = load_ts_from_csv(dummy_csv_path, time_col='timestamp', value_col='value', freq='D')
    #     loaded_ts.summary()
    #     import os
    #     os.remove(dummy_csv_path) # Clean up dummy file
    # except Exception as e:
    #      print(f"\nCSV Loading example failed: {e}")

