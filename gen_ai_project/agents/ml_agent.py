# agents/modeling_agent.py

import logging
import os
import io
import joblib
import traceback
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# --- LangChain Components ---
from langchain_core.prompts import ChatPromptTemplate, SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

# --- Project Imports ---
from .base_agent import BaseAgent # Import the base class

# Logger setup is handled by the BaseAgent's __init__
# We just need to use self.logger

class ModelingAgent(BaseAgent):
    """
    Handles ML model training, evaluation, and saving (classification/regression).
    Inherits common initialization (LLM, workspace, logger) and utilities (like _load_data) from BaseAgent.
    """
    def __init__(self, llm: Any, workspace_dir: str, verbose: bool = False):
        """
        Initializes the ModelingAgent.

        Args:
            llm: The language model instance.
            workspace_dir: The absolute path to the agent's workspace directory.
                           Required for saving models and potentially loading data.
            verbose: If True, enable more detailed logging.
        """
        # Call BaseAgent's init first
        super().__init__(llm=llm, workspace_dir=workspace_dir, verbose=verbose)

        # Specific check: ModelingAgent needs a workspace
        if not self.workspace_dir:
            self.logger.critical("Initialization failed: ModelingAgent requires a valid workspace_dir.")
            raise ValueError("ModelingAgent requires a valid workspace_dir.")

        # Ensure required ML libraries are available (optional check)
        try:
            import sklearn, joblib
            self.logger.debug("Scikit-learn and joblib libraries are available.")
        except ImportError:
             self.logger.critical("Scikit-learn or joblib not found. Please install them (`pip install scikit-learn joblib`).")
             raise ImportError("ModelingAgent requires scikit-learn and joblib.")

        self.logger.info("ModelingAgent specific setup complete.")


    def _parse_modeling_request(self, request: str) -> Optional[Dict[str, Any]]:
        """Uses LLM to parse natural language request into modeling parameters."""
        self.logger.debug(f"Parsing modeling request: '{request[:100]}...'")
        parser = JsonOutputParser()
        # Note: This prompt is crucial and may need refinement based on LLM performance.
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are an expert at interpreting machine learning modeling requests.
Analyze the user's request and extract the following information in JSON format:
- task_type: (string) Infer 'classification' or 'regression' based on the target variable and request (e.g., predict price -> regression, predict category -> classification). REQUIRED.
- target_column: (string) The name of the column to predict. REQUIRED.
- feature_columns: (list of strings or null) Specific columns to use as features. If null or empty, use all other columns (excluding target).
- model_type: (string or null) Specific model requested (e.g., 'LogisticRegression', 'RandomForestClassifier', 'LinearRegression', 'RandomForestRegressor'). If null, use a sensible default based on task_type ('default_classification' or 'default_regression').
- preprocessing: (dict or null) Specify basic preprocessing steps. Example: {"handle_missing": "mean", "scale_numeric": true, "encode_categorical": true}. If null, apply default steps.
- evaluation_metrics: (list of strings or null) Metrics to report (e.g., ['accuracy', 'f1'] for classification, ['rmse', 'r2'] for regression). If null, use defaults.
- output_model_filename: (string or null) Base name for saving the model (e.g., 'salary_predictor'). If null, don't save.

If required parameters like 'target_column' or 'task_type' cannot be inferred, return an error structure like {"error": "Missing required info"}.
Ensure the output is a valid JSON object.
"""),
            HumanMessage(content=f"Parse the following modeling request:\n\n{request}")
        ])

        chain = prompt | self.llm | parser
        try:
            parsed_params = chain.invoke({})
            self.logger.debug(f"LLM parsed modeling parameters: {parsed_params}")
            if not isinstance(parsed_params, dict):
                 self.logger.warning("LLM parsing did not return a dict.")
                 return {"error": "LLM parsing failed to return a dictionary."}
            if "error" in parsed_params:
                 self.logger.warning(f"LLM parsing returned error: {parsed_params['error']}")
                 return parsed_params # Propagate error
            if not parsed_params.get('target_column') or not parsed_params.get('task_type'):
                 self.logger.warning("LLM failed to extract target_column or task_type.")
                 return {"error": "Could not determine target column or task type (classification/regression) from the request."}
            return parsed_params
        except Exception as e:
            self.logger.error(f"Error parsing modeling request with LLM: {e}", exc_info=self.verbose)
            return {"error": f"LLM parsing failed: {e}"}

    # NOTE: _load_data method is now inherited from BaseAgent.

    def _build_preprocessor(self, df: pd.DataFrame, features: List[str], target: str, params: Dict[str, Any]) -> Optional[ColumnTransformer]:
        """Builds a scikit-learn ColumnTransformer for preprocessing."""
        self.logger.debug("Building preprocessor pipeline...")
        prep_params = params.get('preprocessing') or {}
        handle_missing = prep_params.get('handle_missing', 'mean') # mean, median, most_frequent
        scale_numeric = prep_params.get('scale_numeric', True)
        encode_categorical = prep_params.get('encode_categorical', True)
        self.logger.debug(f"Preprocessing settings: missing='{handle_missing}', scale={scale_numeric}, encode={encode_categorical}")

        numeric_features = df[features].select_dtypes(include=np.number).columns.tolist()
        categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
        self.logger.debug(f"Numeric features identified: {numeric_features}")
        self.logger.debug(f"Categorical features identified: {categorical_features}")

        transformers = []

        # Numeric pipeline
        if numeric_features:
            num_steps = []
            if handle_missing in ['mean', 'median', 'most_frequent']: # Added most_frequent
                 self.logger.debug(f"Adding SimpleImputer(strategy='{handle_missing}') for numeric features.")
                 num_steps.append(('imputer', SimpleImputer(strategy=handle_missing)))
            elif handle_missing: # Warn if strategy is invalid but specified
                 self.logger.warning(f"Invalid missing value strategy '{handle_missing}' for numeric features. Skipping imputation.")

            if scale_numeric:
                 self.logger.debug("Adding StandardScaler for numeric features.")
                 num_steps.append(('scaler', StandardScaler()))

            if num_steps:
                 numeric_pipeline = Pipeline(steps=num_steps)
                 transformers.append(('num', numeric_pipeline, numeric_features))

        # Categorical pipeline
        if categorical_features:
            cat_steps = []
            # Impute missing categoricals with a constant placeholder
            self.logger.debug("Adding SimpleImputer(strategy='constant', fill_value='_missing_') for categorical features.")
            cat_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='_missing_')))

            if encode_categorical:
                 # Use OneHotEncoder, ignore unknown values encountered during transform
                 self.logger.debug("Adding OneHotEncoder(handle_unknown='ignore') for categorical features.")
                 cat_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))) # sparse=False often easier

            if cat_steps:
                 categorical_pipeline = Pipeline(steps=cat_steps)
                 transformers.append(('cat', categorical_pipeline, categorical_features))

        if not transformers:
            self.logger.warning("No preprocessing transformers were added. Ensure features exist and preprocessing is enabled if needed.")
            return None # Return None if no transformers are actually created

        try:
            # remainder='passthrough' keeps columns not specified in transformers
            # remainder='drop' would remove them
            preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough', verbose_feature_names_out=False)
            self.logger.debug("ColumnTransformer preprocessor built successfully.")
            return preprocessor
        except Exception as e:
            self.logger.error(f"Error building ColumnTransformer preprocessor: {e}", exc_info=self.verbose)
            return None


    def run_modeling_task(self, data_source: str, request: str) -> str:
        """
        Performs the ML modeling task: load, preprocess, train, evaluate, save.

        Args:
            data_source: Path to the data file (e.g., 'data.csv') or CSV string.
            request: Natural language description of the modeling task.

        Returns:
            A message indicating success (including evaluation metrics and model path) or failure.
        """
        self.logger.info(f"--- Modeling Agent: Running Task ---")
        self.logger.info(f"Data Source: {data_source}")
        self.logger.info(f"Request: {request}")

        # 1. Parse Request
        params = self._parse_modeling_request(request)
        if not params or "error" in params:
            error_msg = f"Error: Could not parse modeling request. {params.get('error', '')}"
            self.logger.error(error_msg)
            return error_msg

        target_col = params['target_column']
        task_type = params['task_type'].lower() # 'classification' or 'regression'
        output_base_filename = params.get('output_model_filename') # Base name for model/preprocessor files

        # 2. Load Data (using inherited method)
        df = self._load_data(data_source)
        if df is None: return f"Error: Could not load data from '{data_source}'" # Error logged in _load_data
        if df.empty: return f"Error: Data loaded from '{data_source}' is empty." # Warning logged in _load_data

        self.logger.debug(f"Data loaded. Shape: {df.shape}, Columns: {df.columns.tolist()}")

        # 3. Identify Features and Target
        if target_col not in df.columns:
            error_msg = f"Error: Target column '{target_col}' not found in data columns: {df.columns.tolist()}"
            self.logger.error(error_msg)
            return error_msg

        feature_cols = params.get('feature_columns')
        if not feature_cols: # Use all columns except target if not specified
            feature_cols = [col for col in df.columns if col != target_col]
            self.logger.debug(f"No specific feature columns requested. Using all columns except target: {feature_cols}")
        else: # Validate specified feature columns
            missing_features = [col for col in feature_cols if col not in df.columns]
            if missing_features:
                error_msg = f"Error: Specified feature columns not found in data: {missing_features}"
                self.logger.error(error_msg)
                return error_msg
            self.logger.debug(f"Using specified feature columns: {feature_cols}")

        if not feature_cols:
             error_msg = "Error: No feature columns identified or specified after processing."
             self.logger.error(error_msg)
             return error_msg

        self.logger.info(f"Target='{target_col}', Features={feature_cols}")
        X = df[feature_cols]
        y = df[target_col]

        # Basic check for target type vs task type
        if task_type == 'classification' and pd.api.types.is_numeric_dtype(y) and y.nunique() > 20: # Heuristic threshold
             self.logger.warning(f"Task is classification, but target '{target_col}' is numeric with {y.nunique()} unique values. Check request parsing or data.")
        if task_type == 'regression' and not pd.api.types.is_numeric_dtype(y):
             self.logger.warning(f"Task is regression, but target '{target_col}' is not numeric ({y.dtype}). Attempting conversion.")
             try:
                 y = pd.to_numeric(y, errors='raise') # Raise error if conversion fails
                 self.logger.info(f"Successfully converted target column '{target_col}' to numeric.")
             except (ValueError, TypeError) as conv_err:
                 error_msg = f"Error: Regression task specified, but target column '{target_col}' could not be converted to numeric: {conv_err}"
                 self.logger.error(error_msg)
                 return error_msg

        # 4. Preprocessing & Data Splitting
        preprocessor = None # Initialize preprocessor
        try:
            preprocessor = self._build_preprocessor(df, feature_cols, target_col, params)
            if preprocessor:
                self.logger.info("Fitting preprocessor on the entire feature set X...")
                # Fit on full X before splitting to avoid data leakage in fit, transform train/test separately later.
                preprocessor.fit(X, y) # Pass y in case preprocessor needs it (though unlikely for these steps)
                self.logger.debug("Preprocessor fitting complete.")
            else:
                 self.logger.info("No preprocessor pipeline was built (or needed based on features/settings).")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if task_type=='classification' and y.nunique() > 1 else None)
            self.logger.info(f"Data split. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            # Apply preprocessing *after* splitting
            if preprocessor:
                 self.logger.info("Transforming train and test sets with preprocessor...")
                 X_train_processed = preprocessor.transform(X_train)
                 X_test_processed = preprocessor.transform(X_test)
                 # Get feature names after transformation (important for some models/analysis)
                 try:
                      # Use verbose_feature_names_out=False in ColumnTransformer for cleaner names
                      feature_names_out = preprocessor.get_feature_names_out()
                      self.logger.debug(f"Processed features ({len(feature_names_out)}): {feature_names_out[:15]}...") # Print first few
                 except Exception as name_err:
                      self.logger.warning(f"Could not get feature names from preprocessor: {name_err}")
                      feature_names_out = None # Fallback
            else:
                 # If no preprocessor, ensure data is in a format models expect (e.g., numpy for sklearn)
                 # Handle potential mixed types if no preprocessing happened
                 try:
                     X_train_processed = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else np.array(X_train)
                     X_test_processed = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else np.array(X_test)
                     feature_names_out = feature_cols # Original features
                     self.logger.debug("Using raw features (converted to numpy array).")
                 except Exception as np_err:
                      error_msg = f"Error converting data to numpy array without preprocessor: {np_err}"
                      self.logger.error(error_msg)
                      return error_msg

        except Exception as e:
            error_msg = f"Error during data splitting or preprocessing: {e}"
            self.logger.error(error_msg, exc_info=self.verbose)
            return error_msg

        # 5. Model Selection & Training
        model_type_req = params.get('model_type')
        model = None
        self.logger.info(f"Selecting model for task '{task_type}' (request: {model_type_req})...")

        try:
            # Define default models
            default_clf = RandomForestClassifier(random_state=42, n_estimators=100)
            default_reg = RandomForestRegressor(random_state=42, n_estimators=100)

            if task_type == 'classification':
                if model_type_req == 'LogisticRegression': model = LogisticRegression(random_state=42, max_iter=1000)
                elif model_type_req == 'RandomForestClassifier': model = default_clf
                else: # Default classification
                    self.logger.info(f"Using default classification model: {default_clf.__class__.__name__}")
                    model = default_clf
            elif task_type == 'regression':
                if model_type_req == 'LinearRegression': model = LinearRegression()
                elif model_type_req == 'RandomForestRegressor': model = default_reg
                else: # Default regression
                    self.logger.info(f"Using default regression model: {default_reg.__class__.__name__}")
                    model = default_reg
            else:
                error_msg = f"Error: Unknown task type '{task_type}' determined."
                self.logger.error(error_msg)
                return error_msg

            self.logger.info(f"Training {model.__class__.__name__}...")
            model.fit(X_train_processed, y_train)
            self.logger.info("Training complete.")

        except Exception as e:
            error_msg = f"Error during model selection or training: {e}"
            self.logger.error(error_msg, exc_info=self.verbose)
            return error_msg

        # 6. Evaluation
        results = {}
        self.logger.info("Evaluating model on the test set...")
        try:
            y_pred = model.predict(X_test_processed)
            requested_metrics = params.get('evaluation_metrics') # Use requested or defaults

            if task_type == 'classification':
                metrics_to_calc = requested_metrics or ['accuracy', 'f1_macro', 'report'] # Default metrics
                self.logger.debug(f"Calculating classification metrics: {metrics_to_calc}")
                if 'accuracy' in metrics_to_calc:
                    results['accuracy'] = accuracy_score(y_test, y_pred)
                # Generate report if needed for f1 or explicitly requested
                if 'report' in metrics_to_calc or 'f1_macro' in metrics_to_calc or 'f1_weighted' in metrics_to_calc:
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    if 'report' in metrics_to_calc:
                        results['classification_report (dict)'] = report # Store full dict report
                    if 'f1_macro' in metrics_to_calc:
                        results['f1_macro_avg'] = report.get('macro avg', {}).get('f1-score', np.nan)
                    if 'f1_weighted' in metrics_to_calc:
                         results['f1_weighted_avg'] = report.get('weighted avg', {}).get('f1-score', np.nan)

            elif task_type == 'regression':
                metrics_to_calc = requested_metrics or ['rmse', 'r2'] # Default metrics
                self.logger.debug(f"Calculating regression metrics: {metrics_to_calc}")
                if 'mse' in metrics_to_calc or 'rmse' in metrics_to_calc:
                    mse = mean_squared_error(y_test, y_pred)
                    if 'mse' in metrics_to_calc: results['MSE'] = mse
                    if 'rmse' in metrics_to_calc: results['RMSE'] = np.sqrt(mse)
                if 'r2' in metrics_to_calc:
                    results['R2_score'] = r2_score(y_test, y_pred)

            self.logger.info(f"Evaluation Metrics: {results}")

        except Exception as e:
            error_msg = f"Model trained, but error during evaluation: {e}"
            self.logger.error(error_msg, exc_info=self.verbose)
            # Return partial success message with error? Or just the error?
            # Let's return the error but mention training was done.
            return error_msg

        # 7. Save Model & Preprocessor (Optional)
        saved_paths = {}
        if output_base_filename:
            self.logger.info(f"Attempting to save model and preprocessor with base name: {output_base_filename}")
            try:
                # Define filenames
                pipeline_filename = f"{output_base_filename}_pipeline.joblib"
                pipeline_path = os.path.join(self.workspace_dir, pipeline_filename)

                # Create a pipeline containing the preprocessor (if exists) and the model
                steps = []
                if preprocessor:
                    steps.append(('preprocessor', preprocessor))
                steps.append(('model', model))
                full_pipeline = Pipeline(steps=steps)

                # Save the entire pipeline
                joblib.dump(full_pipeline, pipeline_path)
                self.logger.info(f"Modeling pipeline (preprocessor + model) saved to {pipeline_path}")
                saved_paths['pipeline_path'] = pipeline_path

                # Optionally save feature names if available
                if feature_names_out is not None:
                     features_filename = f"{output_base_filename}_features.json"
                     features_path = os.path.join(self.workspace_dir, features_filename)
                     save_features_payload = {'feature_names_in': feature_cols, 'feature_names_out': list(feature_names_out), 'target_column': target_col}
                     with open(features_path, 'w') as f:
                          json.dump(save_features_payload, f, indent=4)
                     self.logger.info(f"Feature names saved to {features_path}")
                     saved_paths['feature_names_path'] = features_path

            except Exception as e:
                save_error_msg = f"Error saving modeling pipeline/features: {e}"
                self.logger.error(save_error_msg, exc_info=self.verbose)
                results['saving_error'] = save_error_msg # Add error to results dict

        # 8. Format and Return Result Message
        result_message = f"Successfully completed modeling task '{request}'.\n"
        result_message += "\nEvaluation Metrics:\n"
        for metric, value in results.items():
            if isinstance(value, float):
                result_message += f"- {metric}: {value:.4f}\n"
            elif metric == 'classification_report (dict)':
                 # Provide a summary from the dict report
                 f1_macro = value.get('macro avg', {}).get('f1-score', 'N/A')
                 f1_weighted = value.get('weighted avg', {}).get('f1-score', 'N/A')
                 accuracy = value.get('accuracy', 'N/A') # Accuracy is often outside the nested dict
                 if isinstance(accuracy, float): accuracy = f"{accuracy:.4f}"
                 if isinstance(f1_macro, float): f1_macro = f"{f1_macro:.4f}"
                 if isinstance(f1_weighted, float): f1_weighted = f"{f1_weighted:.4f}"
                 result_message += f"- Accuracy: {accuracy}\n"
                 result_message += f"- F1 Macro Avg: {f1_macro}\n"
                 result_message += f"- F1 Weighted Avg: {f1_weighted}\n"
                 # Optionally add "(Full report dictionary also available in results)"
            else:
                result_message += f"- {metric}: {value}\n" # Handle strings or other types

        if saved_paths:
            result_message += "\nSaved Artifacts:\n"
            for name, path in saved_paths.items():
                result_message += f"- {name}: {os.path.basename(path)} (in workspace)\n" # Show relative path

        self.logger.info("Modeling task finished successfully.")
        return result_message

