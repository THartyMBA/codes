# agents/forecasting_agent.py

import logging
import os
import io
import joblib
import traceback
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.tsa.api as smt # For inferring frequency offset

# --- LangChain Components ---
from langchain_core.prompts import ChatPromptTemplate, SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

# --- Project Imports ---
from .base_agent import BaseAgent # Import the base class

# --- Time Series Utilities ---
# Attempt to import the evaluation function. Adjust path if needed.
# Option 1: If kept in root time_series.py (requires root in PYTHONPATH or careful execution context)
# try:
#     from time_series import calculate_forecast_metrics
# except ImportError:
# Option 2: If moved to utils/time_series_utils.py
try:
    from ..utils.time_series_utils import calculate_forecast_metrics
except ImportError:
    logging.warning("Could not import 'calculate_forecast_metrics' from 'utils.time_series_utils'. Using basic fallback.")
    # Define a basic fallback or skip evaluation if import fails
    def calculate_forecast_metrics(actual, predicted):
        logging.warning("Using basic forecast metrics calculation (RMSE only).")
        if len(actual) != len(predicted): return {'RMSE': np.nan, 'error': 'Length mismatch'}
        try:
            mse = np.mean((np.array(actual) - np.array(predicted))**2)
            return {'RMSE': np.sqrt(mse)}
        except Exception as e:
            return {'RMSE': np.nan, 'error': str(e)}

# Logger setup is handled by the BaseAgent's __init__
# We just need to use self.logger

class ForecastingAgent(BaseAgent):
    """
    Handles Time Series Forecasting tasks using SARIMAX.
    Inherits common initialization (LLM, workspace, logger) and utilities from BaseAgent.
    """
    def __init__(self, llm: Any, workspace_dir: str, verbose: bool = False):
        """
        Initializes the ForecastingAgent.

        Args:
            llm: The language model instance.
            workspace_dir: The absolute path to the agent's workspace directory.
                           Required for saving models/forecasts and potentially loading data.
            verbose: If True, enable more detailed logging.
        """
        # Call BaseAgent's init first
        super().__init__(llm=llm, workspace_dir=workspace_dir, verbose=verbose)

        # Specific check: ForecastingAgent needs a workspace
        if not self.workspace_dir:
            self.logger.critical("Initialization failed: ForecastingAgent requires a valid workspace_dir.")
            raise ValueError("ForecastingAgent requires a valid workspace_dir.")

        # Ensure required libraries are available (optional check)
        try:
            import statsmodels, joblib
            self.logger.debug("Statsmodels and joblib libraries are available.")
        except ImportError:
             self.logger.critical("Statsmodels or joblib not found. Please install them (`pip install statsmodels joblib`).")
             raise ImportError("ForecastingAgent requires statsmodels and joblib.")

        self.logger.info("ForecastingAgent specific setup complete.")


    def _parse_forecasting_request(self, request: str) -> Optional[Dict[str, Any]]:
        """Uses LLM to parse natural language request into forecasting parameters."""
        self.logger.debug(f"Parsing forecasting request: '{request[:100]}...'")
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are an expert at interpreting time series forecasting requests.
Analyze the user's request and extract the following information in JSON format:
- data_source_time_column: (string) The name of the column containing datetime information in the source data. REQUIRED.
- target_column: (string) The name of the column containing the values to forecast. REQUIRED.
- freq: (string or null) The frequency of the time series (e.g., 'D' for daily, 'H' for hourly, 'MS' for month start). If null, attempt to infer. Crucial for modeling.
- forecast_horizon: (integer) How many steps/periods into the future to forecast. REQUIRED.
- model_type: (string, optional) Currently supports 'SARIMAX'. Defaults to 'SARIMAX'.
- order: (list of 3 integers or null, optional) The (p, d, q) order for ARIMA. Example: [1, 1, 1]. If null, use default [1, 1, 1].
- seasonal_order: (list of 4 integers or null, optional) The (P, D, Q, s) seasonal order for SARIMAX. Example: [1, 1, 1, 12]. 's' is the seasonal period (e.g., 7 for daily, 12 for monthly). If null, use default [1, 1, 1, m] where 'm' is inferred from freq.
- output_forecast_filename: (string or null) Base name for saving the forecast results CSV (e.g., 'sales_forecast'). If null, don't save forecast data.
- output_model_filename: (string or null) Base name for saving the trained model (e.g., 'call_volume_model'). If null, don't save model.

If required parameters (time_column, target_column, forecast_horizon) cannot be inferred, return an error structure like {"error": "Missing required info"}.
Ensure the output is a valid JSON object.
"""),
            HumanMessage(content=f"Parse the following forecasting request:\n\n{request}")
        ])

        chain = prompt | self.llm | parser
        try:
            parsed_params = chain.invoke({})
            self.logger.debug(f"LLM parsed forecasting parameters: {parsed_params}")
            if not isinstance(parsed_params, dict):
                 self.logger.warning("LLM parsing did not return a dict.")
                 return {"error": "LLM parsing failed (not dict)."}
            if "error" in parsed_params:
                 self.logger.warning(f"LLM parsing returned error: {parsed_params['error']}")
                 return parsed_params
            if not all(k in parsed_params for k in ['data_source_time_column', 'target_column', 'forecast_horizon']):
                 self.logger.warning("LLM failed to extract required parameters: time_column, target_column, or forecast_horizon.")
                 return {"error": "Missing required info: time_column, target_column, or forecast_horizon."}
            if not isinstance(parsed_params['forecast_horizon'], int) or parsed_params['forecast_horizon'] <= 0:
                 self.logger.warning(f"Invalid forecast_horizon: {parsed_params['forecast_horizon']}. Must be positive integer.")
                 return {"error": "forecast_horizon must be a positive integer."}
            # Basic validation for order/seasonal_order if provided
            if parsed_params.get('order') and (not isinstance(parsed_params['order'], list) or len(parsed_params['order']) != 3 or not all(isinstance(i, int) for i in parsed_params['order'])):
                 self.logger.warning(f"Invalid 'order' format: {parsed_params['order']}. Must be list of 3 integers.")
                 return {"error": "Invalid 'order' format. Must be a list of 3 integers (p, d, q)."}
            if parsed_params.get('seasonal_order') and (not isinstance(parsed_params['seasonal_order'], list) or len(parsed_params['seasonal_order']) != 4 or not all(isinstance(i, int) for i in parsed_params['seasonal_order'])):
                 self.logger.warning(f"Invalid 'seasonal_order' format: {parsed_params['seasonal_order']}. Must be list of 4 integers.")
                 return {"error": "Invalid 'seasonal_order' format. Must be a list of 4 integers (P, D, Q, s)."}

            return parsed_params
        except Exception as e:
            self.logger.error(f"Error parsing forecasting request with LLM: {e}", exc_info=self.verbose)
            return {"error": f"LLM parsing failed: {e}"}

    # NOTE: _load_data method is inherited from BaseAgent.
    # We need a specific loader for time series data handling.

    def _load_ts_data(self, data_source: str, time_col: str, value_col: str, freq: Optional[str], datetime_format: Optional[str] = None) -> Optional[pd.Series]:
        """
        Loads time series data using BaseAgent._load_data, then processes it:
        sets DatetimeIndex, ensures frequency, handles missing values.

        Args:
            data_source: Path/CSV string passed to _load_data.
            time_col: Name of the datetime column.
            value_col: Name of the column with values to forecast.
            freq: Pandas frequency string (e.g., 'D', 'H', 'MS'). If None, attempts inference.
            datetime_format: Optional format string for pd.to_datetime.

        Returns:
            A pandas Series with DatetimeIndex and frequency, or None if processing fails.
        """
        self.logger.info(f"Loading and processing time series data. Time col='{time_col}', Value col='{value_col}', Freq='{freq}'")
        df = self._load_data(data_source) # Use inherited method

        if df is None:
            # Error already logged by _load_data
            return None
        if df.empty:
             self.logger.error("Cannot process empty DataFrame for time series.")
             return None

        try:
            if time_col not in df.columns:
                raise ValueError(f"Time column '{time_col}' not found in data columns: {df.columns.tolist()}")
            if value_col not in df.columns:
                raise ValueError(f"Value column '{value_col}' not found in data columns: {df.columns.tolist()}")

            # Convert to datetime and set index
            self.logger.debug(f"Converting '{time_col}' to datetime...")
            if datetime_format:
                df[time_col] = pd.to_datetime(df[time_col], format=datetime_format, errors='coerce')
            else:
                # Try inferring format, coerce errors to NaT
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

            # Check for conversion errors
            if df[time_col].isnull().any():
                 num_failed = df[time_col].isnull().sum()
                 self.logger.warning(f"{num_failed} values in '{time_col}' failed datetime conversion and became NaT.")
                 # Option: Drop rows with NaT index or raise error? Let's drop for now.
                 df = df.dropna(subset=[time_col])
                 if df.empty:
                      raise ValueError(f"All values in '{time_col}' failed datetime conversion.")

            df = df.set_index(time_col)
            df = df[[value_col]] # Keep only the target column
            df = df.sort_index()
            self.logger.debug(f"Data indexed by '{time_col}' and sorted.")

            # Handle Frequency
            inferred_freq = pd.infer_freq(df.index)
            target_freq = freq
            self.logger.debug(f"Specified freq='{freq}', Inferred freq='{inferred_freq}'")

            if target_freq:
                 self.logger.info(f"Resampling data to specified frequency: {target_freq}")
                 # Use asfreq, decide how to handle missing values introduced by upsampling (e.g., ffill)
                 # Using ffill assumes the value persists until the next known value
                 df = df.asfreq(target_freq, method='ffill')
            elif inferred_freq:
                 target_freq = inferred_freq
                 df = df.asfreq(target_freq) # Ensure frequency is explicitly set
                 self.logger.info(f"Using inferred frequency: {target_freq}")
            else:
                 # Attempt to make regular if possible, otherwise raise error
                 self.logger.warning("Could not directly infer frequency. Checking median difference.")
                 median_diff = df.index.to_series().diff().median()
                 if pd.notna(median_diff):
                      # Use statsmodels to try and get a frequency string
                      try:
                          inferred_offset = smt.frequencies.to_offset(median_diff)
                          if inferred_offset:
                               target_freq = inferred_offset.freqstr
                               self.logger.warning(f"Attempting to use frequency based on median difference: {target_freq}")
                               df = df.asfreq(target_freq, method='ffill') # Try setting it, fill gaps
                          else:
                               raise ValueError("Could not determine frequency offset from median difference.")
                      except Exception as freq_err:
                           raise ValueError(f"Could not infer frequency and no frequency specified. Error: {freq_err}") from freq_err
                 else:
                      raise ValueError("Could not infer frequency (index difference inconsistent) and no frequency specified.")

            if df.empty:
                 raise ValueError(f"DataFrame became empty after setting frequency '{target_freq}'. Check data range and frequency.")

            # Handle missing values (simple forward fill then backward fill)
            initial_nan = df[value_col].isna().sum()
            if initial_nan > 0:
                self.logger.info(f"Filling {initial_nan} missing values in '{value_col}' using ffill/bfill.")
                df[value_col] = df[value_col].ffill().bfill()

            if df[value_col].isna().any():
                 final_nan = df[value_col].isna().sum()
                 raise ValueError(f"Data still contains {final_nan} NaNs in '{value_col}' after filling. Cannot proceed.")

            # Return just the Series, indexed by time, with frequency
            ts_series = df[value_col]
            # Ensure frequency is attached to the Series index (asfreq should do this, but double-check)
            if not ts_series.index.freq:
                 ts_series.index.freq = target_freq
            self.logger.info(f"Time series processed. Length: {len(ts_series)}, Freq: {ts_series.index.freqstr}")
            return ts_series

        except Exception as e:
            self.logger.error(f"Error loading or processing time series data from '{data_source}': {e}", exc_info=self.verbose)
            return None

    def _get_seasonal_period(self, freq_str: Optional[str]) -> int:
        """Infer seasonal period 'm' from frequency string."""
        if not freq_str:
            self.logger.warning("Frequency string is None, cannot determine seasonal period. Defaulting to 1.")
            return 1
        # Use statsmodels offset object for more reliable parsing
        try:
            offset = smt.frequencies.to_offset(freq_str)
            base_name = offset.name.split('-')[0] # Get base frequency like 'D', 'M', 'H'

            if base_name == 'D': return 7
            if base_name == 'W': return 52 # Approximation
            if base_name in ['M', 'MS']: return 12
            if base_name in ['Q', 'QS']: return 4
            if base_name in ['A', 'AS', 'Y', 'YS']: return 1 # Yearly
            if base_name == 'H': return 24
            if base_name in ['T', 'min']: return 60
            if base_name == 'S': return 60
            if base_name == 'B': return 5 # Business days -> weekly seasonality

            self.logger.warning(f"Could not determine standard seasonal period for frequency '{freq_str}' (base='{base_name}'). Defaulting to 1.")
            return 1 # Default if unknown
        except Exception as e:
             self.logger.warning(f"Error parsing frequency '{freq_str}' for seasonal period: {e}. Defaulting to 1.")
             return 1


    def _train_and_forecast(self, ts_series: pd.Series, params: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[Any], Optional[Dict]]:
        """Trains SARIMAX model and generates forecast."""
        horizon = params['forecast_horizon']
        order = tuple(params.get('order', [1, 1, 1])) # Default ARIMA order, ensure tuple
        seasonal_order_req = params.get('seasonal_order')
        freq = ts_series.index.freqstr # Get freq from the series index

        if not freq:
             self.logger.error("Time series frequency is missing after loading. Cannot determine seasonal period.")
             return None, None, {"error": "Time series frequency missing."}

        m = self._get_seasonal_period(freq)

        # Default seasonal order if not provided
        if seasonal_order_req:
             seasonal_order = tuple(seasonal_order_req) # Ensure tuple
             # Warn if user-provided 's' differs from inferred 'm'
             if seasonal_order[3] != m:
                  self.logger.warning(f"Provided seasonal period s={seasonal_order[3]} differs from inferred period m={m} based on frequency '{freq}'. Using provided s={seasonal_order[3]}.")
                  # No need to update m here, SARIMAX uses seasonal_order[3] directly
             else:
                  self.logger.debug(f"Using provided seasonal order with period s={seasonal_order[3]}.")
        else:
             seasonal_order = (1, 1, 1, m) # Default seasonal order using inferred m
             self.logger.debug(f"Using default seasonal order with inferred period m={m}.")

        self.logger.info(f"Using SARIMAX order={order}, seasonal_order={seasonal_order}")

        # --- Evaluation Step (Train/Test Split) ---
        eval_metrics = {}
        # Use a test set at least as long as horizon or seasonality period, whichever is larger
        test_size_heuristic = max(horizon, seasonal_order[3] if seasonal_order[3] > 0 else 1, 10) # Ensure > 0
        if len(ts_series) > 2 * test_size_heuristic: # Only evaluate if enough data for train/test
            self.logger.info(f"Performing evaluation on hold-out set (size={test_size_heuristic})...")
            train_ts = ts_series[:-test_size_heuristic]
            test_ts = ts_series[-test_size_heuristic:]
            try:
                # Fit model on training data
                eval_model = SARIMAX(train_ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                eval_model_fit = eval_model.fit(disp=False) # Suppress convergence output
                # Forecast for the length of the test set
                eval_forecast_obj = eval_model_fit.get_forecast(steps=len(test_ts))
                eval_preds = eval_forecast_obj.predicted_mean
                # Use the imported or fallback metric calculator
                eval_metrics = calculate_forecast_metrics(test_ts, eval_preds)
                self.logger.info(f"Evaluation Metrics (on hold-out set): {eval_metrics}")
            except Exception as e:
                self.logger.warning(f"Error during evaluation step: {e}. Skipping evaluation.", exc_info=self.verbose)
                eval_metrics = {"error": f"Evaluation failed: {e}"}
        else:
            self.logger.warning(f"Not enough data (length {len(ts_series)}) for robust evaluation split (need > {2 * test_size_heuristic}). Skipping evaluation.")
            eval_metrics = {"info": "Skipped evaluation due to insufficient data."}


        # --- Final Model Training and Forecasting ---
        self.logger.info(f"Training final model on full data (size={len(ts_series)})...")
        try:
            model = SARIMAX(ts_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False) # Suppress convergence output during final fit
            self.logger.info("Final model training complete.")
            self.logger.debug(f"Final model summary:\n{model_fit.summary()}") # Log summary at debug level

            # Generate forecast
            self.logger.info(f"Generating forecast for {horizon} steps...")
            forecast_obj = model_fit.get_forecast(steps=horizon)
            # Get mean forecast and confidence intervals
            forecast_df = forecast_obj.summary_frame(alpha=0.05) # alpha=0.05 for 95% CI
            # Select and rename relevant columns for clarity
            forecast_df = forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']]
            forecast_df.rename(columns={'mean': 'predicted_value', 'mean_ci_lower': 'lower_ci_95', 'mean_ci_upper': 'upper_ci_95'}, inplace=True)

            self.logger.info(f"Forecast generated successfully.")
            return forecast_df, model_fit, eval_metrics

        except Exception as e:
            self.logger.error(f"Error during final model training or forecasting: {e}", exc_info=self.verbose)
            # Return eval metrics even if final forecast fails, but indicate forecast failure
            return None, None, eval_metrics


    def run_forecasting_task(self, data_source: str, request: str) -> str:
        """
        Performs the Time Series Forecasting task.

        Args:
            data_source: Path to the data file (e.g., 'calls.csv') or CSV string.
            request: Natural language description of the forecasting task.

        Returns:
            A message indicating success (including forecast summary, eval metrics, paths) or failure.
        """
        self.logger.info(f"--- Forecasting Agent: Running Task ---")
        self.logger.info(f"Data Source: {data_source}")
        self.logger.info(f"Request: {request}")

        # 1. Parse Request
        params = self._parse_forecasting_request(request)
        if not params or "error" in params:
            error_msg = f"Error: Could not parse forecasting request. {params.get('error', '')}"
            self.logger.error(error_msg)
            return error_msg

        time_col = params['data_source_time_column']
        target_col = params['target_column']
        freq = params.get('freq') # Can be None initially
        output_forecast_filename = params.get('output_forecast_filename')
        output_model_filename = params.get('output_model_filename')

        # 2. Load and Process Time Series Data
        ts_series = self._load_ts_data(data_source, time_col, target_col, freq)
        if ts_series is None:
            # Error logged in _load_ts_data
            return f"Error: Could not load or process time series data from '{data_source}'."
        # Empty check already handled in _load_ts_data raising error

        # 3. Train and Forecast (includes evaluation)
        forecast_df, fitted_model, eval_metrics = self._train_and_forecast(ts_series, params)

        # Check if forecasting step failed
        if forecast_df is None or fitted_model is None:
            eval_info = f"\nEvaluation Metrics (if attempted): {eval_metrics}" if eval_metrics else ""
            error_msg = f"Error: Failed to train model or generate forecast.{eval_info}"
            self.logger.error(error_msg) # Error already logged in _train_and_forecast
            return error_msg

        # 4. Save Results (Optional)
        saved_paths = {}
        if output_forecast_filename:
            self.logger.info(f"Attempting to save forecast data with base name: {output_forecast_filename}")
            try:
                forecast_filename_csv = f"{output_forecast_filename}_forecast.csv"
                forecast_path = os.path.join(self.workspace_dir, forecast_filename_csv)
                forecast_df.to_csv(forecast_path)
                self.logger.info(f"Forecast data saved to {forecast_path}")
                saved_paths['forecast_data_path'] = forecast_path
            except Exception as e:
                save_error_msg = f"Error saving forecast data: {e}"
                self.logger.error(save_error_msg, exc_info=self.verbose)
                eval_metrics['forecast_saving_error'] = save_error_msg # Add error info

        if output_model_filename:
            self.logger.info(f"Attempting to save trained model with base name: {output_model_filename}")
            try:
                model_filename_joblib = f"{output_model_filename}_model.joblib"
                model_path = os.path.join(self.workspace_dir, model_filename_joblib)
                # Save the fitted SARIMAX results object (contains model params, etc.)
                joblib.dump(fitted_model, model_path)
                self.logger.info(f"Trained SARIMAX model results saved to {model_path}")
                saved_paths['trained_model_path'] = model_path
            except Exception as e:
                save_error_msg = f"Error saving trained model: {e}"
                self.logger.error(save_error_msg, exc_info=self.verbose)
                eval_metrics['model_saving_error'] = save_error_msg # Add error info

        # 5. Format and Return Result Message
        result_message = f"Successfully completed forecasting task '{request}'.\n"
        result_message += f"Forecast Horizon: {params['forecast_horizon']} steps.\n"

        result_message += "\nEvaluation Metrics (on hold-out set, if performed):\n"
        if eval_metrics:
            for metric, value in eval_metrics.items():
                 if isinstance(value, float): result_message += f"- {metric}: {value:.4f}\n"
                 else: result_message += f"- {metric}: {value}\n"
        else:
             # Should have info/error from eval_metrics dict if not performed
             result_message += "- Evaluation metrics not available.\n"

        result_message += "\nForecast Summary (first 5 steps):\n"
        result_message += forecast_df.head().to_string() + "\n"

        if saved_paths:
            result_message += "\nSaved Artifacts:\n"
            for name, path in saved_paths.items():
                result_message += f"- {name}: {os.path.basename(path)} (in workspace)\n"

        self.logger.info("Forecasting task finished successfully.")
        return result_message
