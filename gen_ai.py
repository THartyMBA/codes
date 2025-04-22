import os
import io # For handling string data as files
import json # For parsing LLM output
import joblib # For saving/loading ML models
import traceback # For detailed error logging
from typing import List, Dict, Any, Optional, Tuple

# --- Database Interaction ---
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# --- Core LangChain/LangGraph ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool, BaseTool
from langchain.agents import AgentType, initialize_agent, AgentExecutor

# --- File Management ---
from langchain.agents.agent_toolkits import FileManagementToolkit

# --- Visualization ---
import pandas as pd
import numpy as np # For numeric operations in preprocessing/evaluation
import matplotlib.pyplot as plt
import seaborn as sns

# --- Machine Learning ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# --- Time Series Forecasting ---
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller # For potential future use (e.g., determining 'd')
# Import the evaluation function if it's defined in time_series.py
try:
    # Assuming time_series.py is in the same directory or accessible via PYTHONPATH
    from time_series import calculate_forecast_metrics
except ImportError:
    print("Warning: Could not import 'calculate_forecast_metrics' from time_series.py. Forecast evaluation might be limited.")
    # Define a basic fallback or skip evaluation if import fails
    def calculate_forecast_metrics(actual, predicted):
        print("Warning: Using basic forecast metrics calculation (RMSE only).")
        if len(actual) != len(predicted): return {'RMSE': np.nan, 'error': 'Length mismatch'}
        try:
            mse = np.mean((actual - predicted)**2)
            return {'RMSE': np.sqrt(mse)}
        except Exception as e:
            return {'RMSE': np.nan, 'error': str(e)}



# --- Agent State (Example for potential graph expansion) ---
from typing import TypedDict, Annotated, Sequence
import operator
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], operator.add]
#     # Add other state elements if building a more complex graph

# --- SQL Agent Class ---
class SQLAgent:
    """
    An agent specialized in interacting with a SQL database.
    (Code from previous step - assuming it's correct and complete)
    """
    def __init__(self, llm, db_uri: str, verbose: bool = True):
        self.llm = llm
        self.db_uri = db_uri
        self.verbose = verbose
        self.db = self._connect_db()
        self.agent_executor = self._create_sql_agent()

    def _connect_db(self) -> SQLDatabase:
        if not self.db_uri:
            raise ValueError("Database URI cannot be empty.")
        try:
            engine = create_engine(self.db_uri)
            # Include specific tables if needed, otherwise it tries to infer
            # db = SQLDatabase(engine, include_tables=['employees', 'departments'])
            db = SQLDatabase(engine)
            print(f"Successfully connected to database: {db.dialect}")
            usable_tables = db.get_usable_table_names()
            print(f"Usable Tables: {usable_tables}")
            if not usable_tables:
                 print("Warning: No usable tables found by SQLDatabase. Agent may struggle.")
            # print(f"Sample table info (employees): {db.get_table_info(['employees'])}") # Example inspection
            return db
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database at {self.db_uri}: {e}")

    def _create_sql_agent(self) -> AgentExecutor:
        try:
            # Use the community toolkit if available and preferred
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            # Context provided to the agent can be crucial
            agent_context = """
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
            Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
            You can order the results by a relevant column to return the most interesting examples in the database.
            Never query for all the columns from a specific table, only ask for the relevant columns given the question.
            You have access to tools for interacting with the database.
            Only use the below tools. Only use the information returned by the below tools to construct your final answer.
            You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

            If the question does not seem related to the database, just return "I don't know" as the answer.

            Okay, begin!
            """.format(dialect=self.db.dialect, top_k=10) # Example context, adjust top_k as needed

            return create_sql_agent(
                llm=self.llm,
                toolkit=toolkit, # Pass the toolkit explicitly
                # db=self.db, # Can often be inferred from toolkit
                agent_type="openai-tools", # Or AgentType.OPENAI_FUNCTIONS, AgentType.ZERO_SHOT_REACT_DESCRIPTION
                # input_variables=["input", "agent_scratchpad", "db_info"], # Adjust based on agent type/prompt
                # agent_scratchpad=lambda x: format_to_openai_function_messages(x["intermediate_steps"]), # Example for function calling
                verbose=self.verbose,
                handle_parsing_errors=True,
                # prefix=agent_context, # Add context/instructions if needed
                agent_executor_kwargs={"return_intermediate_steps": True}
            )
        except Exception as e:
            print(f"Error creating SQL agent: {e}. Ensure the LLM and toolkit are compatible.")
            raise

    def run(self, query: str) -> Dict[str, Any]:
        print(f"\n--- Running SQL Agent Query ---")
        print(f"Natural Language Query: {query}")
        generated_sql = "Could not extract SQL."
        try:
            # Ensure input matches what the specific agent expects (often a dict)
            response = self.agent_executor.invoke({"input": query})

            if 'intermediate_steps' in response and response['intermediate_steps']:
                for step in reversed(response['intermediate_steps']):
                     action, observation = step # Assuming step is (AgentAction, Observation)
                     # Check action.tool and action.tool_input
                     # Tool name might be 'sql_db_query', 'query-sql', 'sql_db_query_checker', etc.
                     # Check the logs if unsure about the exact tool name being used
                     tool_name = getattr(action, 'tool', '').lower()
                     tool_input = getattr(action, 'tool_input', None)

                     if 'sql' in tool_name and tool_input:
                         if isinstance(tool_input, str):
                             generated_sql = tool_input
                             break
                         elif isinstance(tool_input, dict) and 'query' in tool_input:
                             generated_sql = tool_input['query']
                             break

            # Clean up SQL potentially wrapped in markdown
            if generated_sql.startswith("```sql"):
                generated_sql = generated_sql.split("```sql\n")[1].split("\n```")[0]

            print(f"Generated SQL (best effort): {generated_sql}")
            print(f"Query Result: {response.get('output', 'N/A')}")
            print(f"--- SQL Agent Finished ---")

            return {
                "result": response.get('output', 'No result found.'),
                "generated_sql": generated_sql,
                "error": None
            }
        except Exception as e:
            print(f"Error during SQL agent execution: {e}")
            # Try to get partial info if available
            output = None
            if 'response' in locals() and isinstance(response, dict):
                output = response.get('output')
            return {
                "result": output,
                "generated_sql": generated_sql,
                "error": str(e)
            }

# --- Visualization Agent Logic ---
class VisualizationAgent:
    """
    Handles the creation of visualizations based on data and requests.
    This class doesn't run a persistent agent loop, but provides the logic
    for the visualization tool used by the Supervisor.
    """
    def __init__(self, llm, workspace_dir: str):
        self.llm = llm
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
        print(f"Visualization Agent using workspace: {self.workspace_dir}")

    def _parse_visualization_request(self, request: str) -> Optional[Dict[str, Any]]:
        """Uses LLM to parse natural language request into plotting parameters."""
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are an expert at interpreting visualization requests and extracting parameters.
Analyze the user's request and extract the following information in JSON format:
- plot_type: (string) The type of plot requested (e.g., 'scatter', 'line', 'bar', 'hist', 'box', 'heatmap'). Choose the most appropriate based on the request.
- x_col: (string or null) The column name for the x-axis.
- y_col: (string or null) The column name for the y-axis.
- color_col: (string or null) The column name to use for coloring (hue).
- size_col: (string or null) The column name to use for sizing points (scatter plots).
- text_col: (string or null) The column name for text labels on points (if requested).
- aggregate_func: (string or null) For bar charts, the aggregation function if needed (e.g., 'mean', 'sum', 'count'). If 'count' is implied (e.g., "count of items per category"), set this to 'count' and y_col might be null.
- title: (string or null) The desired title for the plot.
- xlabel: (string or null) The desired label for the x-axis.
- ylabel: (string or null) The desired label for the y-axis.

If a parameter is not mentioned or cannot be inferred, use null.
Ensure the output is a valid JSON object.
"""),
            HumanMessage(content=f"Parse the following visualization request:\n\n{request}")
        ])

        chain = prompt | self.llm | parser
        try:
            parsed_params = chain.invoke({})
            print(f"Parsed visualization parameters: {parsed_params}")
            # Basic validation
            if not isinstance(parsed_params, dict) or 'plot_type' not in parsed_params:
                 print("Warning: LLM parsing failed to return valid JSON with plot_type.")
                 return None
            return parsed_params
        except Exception as e:
            print(f"Error parsing visualization request with LLM: {e}")
            return None

    def _load_data(self, data_source: str) -> Optional[pd.DataFrame]:
        """Loads data from a file path or a CSV string."""
        try:
            # Check if it's a file path that exists
            potential_path = os.path.join(self.workspace_dir, data_source) # Check relative to workspace first
            if not os.path.exists(potential_path):
                 potential_path = data_source # Check if it's an absolute path

            if os.path.exists(potential_path) and potential_path.lower().endswith('.csv'):
                print(f"Loading data from CSV file: {potential_path}")
                return pd.read_csv(potential_path)
            elif os.path.exists(potential_path) and potential_path.lower().endswith(('.xls', '.xlsx')):
                 print(f"Loading data from Excel file: {potential_path}")
                 return pd.read_excel(potential_path)
            else:
                # Assume it's CSV data passed as a string
                print("Attempting to load data from string (assuming CSV format).")
                # Simple check if it looks like CSV
                if ',' in data_source and '\n' in data_source:
                    data_io = io.StringIO(data_source)
                    return pd.read_csv(data_io)
                else:
                    print(f"Error: Data source '{data_source}' is not a recognized file path or CSV string.")
                    return None
        except Exception as e:
            print(f"Error loading data from source '{data_source}': {e}")
            return None

    def generate_plot(self, data_source: str, request: str, output_filename: str) -> str:
        """
        Generates a plot based on the request and saves it.

        Args:
            data_source: Path to the data file (e.g., 'data.csv') relative to the workspace
                         or the raw data as a CSV string.
            request: Natural language description of the plot.
                     (e.g., "Create a scatter plot of salary vs experience, colored by department")
            output_filename: The desired name for the output plot file (e.g., 'plot.png').
                             Should end with a valid image extension (.png, .jpg, .svg).

        Returns:
            A message indicating success (including the output path) or failure.
        """
        print(f"\n--- Generating Visualization ---")
        print(f"Data Source: {data_source}")
        print(f"Request: {request}")
        print(f"Output Filename: {output_filename}")

        # 1. Parse Request
        params = self._parse_visualization_request(request)
        if not params:
            return f"Error: Could not parse the visualization request: '{request}'"

        # 2. Load Data
        df = self._load_data(data_source)
        if df is None:
            return f"Error: Could not load data from source: '{data_source}'"
        if df.empty:
            return f"Error: Data loaded from '{data_source}' is empty."

        print(f"Data loaded successfully. Columns: {df.columns.tolist()}")

        # 3. Validate Columns
        required_cols = [params.get('x_col'), params.get('y_col'), params.get('color_col'), params.get('size_col'), params.get('text_col')]
        for col in required_cols:
            if col and col not in df.columns:
                return f"Error: Column '{col}' requested for plotting not found in the data."

        # 4. Generate Plot
        plt.figure(figsize=(10, 6)) # Standard figure size
        plot_type = params.get('plot_type', '').lower()
        sns.set_theme(style="whitegrid") # Apply a nice theme

        try:
            if plot_type == 'scatter':
                sns.scatterplot(data=df, x=params.get('x_col'), y=params.get('y_col'),
                                hue=params.get('color_col'), size=params.get('size_col'))
            elif plot_type == 'line':
                # Line plots often assume sorted x-axis (like time)
                x_col = params.get('x_col')
                if x_col and pd.api.types.is_datetime64_any_dtype(df[x_col]):
                     df = df.sort_values(by=x_col)
                elif x_col and pd.api.types.is_numeric_dtype(df[x_col]):
                     df = df.sort_values(by=x_col)
                sns.lineplot(data=df, x=x_col, y=params.get('y_col'), hue=params.get('color_col'))
            elif plot_type == 'bar':
                x_col = params.get('x_col')
                y_col = params.get('y_col')
                hue_col = params.get('color_col')
                agg_func = params.get('aggregate_func')

                if agg_func == 'count' and x_col:
                     # Count occurrences of x_col categories
                     sns.countplot(data=df, x=x_col, hue=hue_col)
                     params['ylabel'] = params.get('ylabel') or 'Count' # Auto-set ylabel if countplot
                elif x_col and y_col:
                     # Bar plot aggregating y_col over x_col categories
                     estimator = agg_func if agg_func in ['mean', 'median', 'sum'] else 'mean' # Default to mean
                     sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, estimator=estimator)
                else:
                    return f"Error: Bar plot requires at least 'x_col', and either 'y_col' or aggregate_func='count'."

            elif plot_type == 'hist' or plot_type == 'histogram':
                if not params.get('x_col'): return "Error: Histogram requires 'x_col'."
                sns.histplot(data=df, x=params.get('x_col'), hue=params.get('color_col'), kde=True) # Add KDE curve
            elif plot_type == 'box' or plot_type == 'boxplot':
                if not params.get('x_col') and not params.get('y_col'): return "Error: Box plot requires 'x_col' or 'y_col'."
                # Seaborn often uses y for numeric, x for category, but can swap
                sns.boxplot(data=df, x=params.get('x_col'), y=params.get('y_col'), hue=params.get('color_col'))
            elif plot_type == 'heatmap':
                 # Heatmaps typically need data in a matrix format or require pivoting
                 # This is a simplified version assuming numeric columns
                 numeric_df = df.select_dtypes(include=np.number)
                 if numeric_df.shape[1] < 2:
                      return "Error: Heatmap requires at least two numeric columns in the data."
                 sns.heatmap(numeric_df.corr(), annot=True, cmap='viridis') # Example: Correlation heatmap
                 params['title'] = params.get('title') or 'Correlation Heatmap'
            else:
                return f"Error: Unsupported plot type requested: '{plot_type}'. Supported types: scatter, line, bar, hist, box, heatmap."

            # 5. Customize Plot
            plt.title(params.get('title') or f"{plot_type.capitalize()} Plot") # Default title
            if params.get('xlabel'): plt.xlabel(params.get('xlabel'))
            if params.get('ylabel'): plt.ylabel(params.get('ylabel'))
            plt.tight_layout()

            # 6. Save Plot
            # Ensure output filename has a valid extension
            valid_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
            if not any(output_filename.lower().endswith(ext) for ext in valid_extensions):
                output_filename += '.png' # Default to PNG
                print(f"Warning: Output filename lacked valid extension, defaulting to {output_filename}")

            output_path = os.path.join(self.workspace_dir, output_filename)
            plt.savefig(output_path)
            plt.close() # Close the figure to free memory

            print(f"Visualization saved successfully to: {output_path}")
            print(f"--- Visualization Finished ---")
            return f"Successfully created visualization and saved it to '{output_path}'"

        except Exception as e:
            plt.close() # Ensure plot is closed even on error
            print(f"Error during plot generation: {e}")
            return f"Error generating plot: {e}"


# --- Modeling Agent Logic ---
class ModelingAgent:
    """
    Handles ML model training, evaluation, and saving.
    """
    def __init__(self, llm, workspace_dir: str):
        self.llm = llm
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
        print(f"Modeling Agent using workspace: {self.workspace_dir}")

    def _parse_modeling_request(self, request: str) -> Optional[Dict[str, Any]]:
        """Uses LLM to parse natural language request into modeling parameters."""
        parser = JsonOutputParser()
        # Note: This prompt is crucial and may need refinement based on LLM performance.
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are an expert at interpreting machine learning modeling requests.
Analyze the user's request and extract the following information in JSON format:
- task_type: (string) Infer 'classification' or 'regression' based on the target variable and request (e.g., predict price -> regression, predict category -> classification).
- target_column: (string) The name of the column to predict. THIS IS REQUIRED.
- feature_columns: (list of strings or null) Specific columns to use as features. If null or empty, use all other columns (excluding target).
- model_type: (string or null) Specific model requested (e.g., 'LogisticRegression', 'RandomForestClassifier', 'LinearRegression', 'RandomForestRegressor'). If null, use a sensible default based on task_type ('default_classification' or 'default_regression').
- preprocessing: (dict or null) Specify basic preprocessing steps. Example: {"handle_missing": "mean", "scale_numeric": true, "encode_categorical": true}. If null, apply default steps.
- evaluation_metrics: (list of strings or null) Metrics to report (e.g., ['accuracy', 'f1'] for classification, ['rmse', 'r2'] for regression). If null, use defaults.
- output_model_filename: (string or null) Base name for saving the model (e.g., 'salary_predictor'). If null, don't save.

If a required parameter like 'target_column' or 'task_type' cannot be inferred, return an error structure like {"error": "Missing required info"}.
Ensure the output is a valid JSON object.
"""),
            HumanMessage(content=f"Parse the following modeling request:\n\n{request}")
        ])

        chain = prompt | self.llm | parser
        try:
            parsed_params = chain.invoke({})
            print(f"Modeling Agent: Parsed Params: {parsed_params}")
            if not isinstance(parsed_params, dict):
                 print("Warning: LLM parsing did not return a dict.")
                 return {"error": "LLM parsing failed to return a dictionary."}
            if "error" in parsed_params:
                 print(f"Warning: LLM parsing returned error: {parsed_params['error']}")
                 return parsed_params # Propagate error
            if not parsed_params.get('target_column') or not parsed_params.get('task_type'):
                 print("Warning: LLM failed to extract target_column or task_type.")
                 return {"error": "Could not determine target column or task type (classification/regression) from the request."}
            return parsed_params
        except Exception as e:
            print(f"Error parsing modeling request with LLM: {e}\n{traceback.format_exc()}")
            return {"error": f"LLM parsing failed: {e}"}

    def _load_data(self, data_source: str) -> Optional[pd.DataFrame]:
        """Loads data (reusing logic similar to VisualizationAgent)."""
        path = data_source
        if not os.path.isabs(path): path = os.path.join(self.workspace_dir, data_source)
        try:
            if os.path.exists(path) and path.lower().endswith('.csv'): return pd.read_csv(path)
            if os.path.exists(path) and path.lower().endswith(('.xls', '.xlsx')): return pd.read_excel(path)
            if ',' in data_source and '\n' in data_source: return pd.read_csv(io.StringIO(data_source))
            print(f"Modeling Agent Load Error: Cannot read {data_source}")
            return None
        except Exception as e: print(f"Modeling Agent Load Error: {e}"); return None

    def _build_preprocessor(self, df: pd.DataFrame, features: List[str], target: str, params: Dict[str, Any]) -> Optional[ColumnTransformer]:
        """Builds a scikit-learn ColumnTransformer for preprocessing."""
        prep_params = params.get('preprocessing') or {}
        handle_missing = prep_params.get('handle_missing', 'mean') # mean, median, most_frequent
        scale_numeric = prep_params.get('scale_numeric', True)
        encode_categorical = prep_params.get('encode_categorical', True)

        numeric_features = df[features].select_dtypes(include=np.number).columns.tolist()
        categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()

        transformers = []

        # Numeric pipeline
        if numeric_features:
            num_steps = []
            if handle_missing in ['mean', 'median']:
                 num_steps.append(('imputer', SimpleImputer(strategy=handle_missing)))
            if scale_numeric:
                 num_steps.append(('scaler', StandardScaler()))
            if num_steps:
                 numeric_pipeline = Pipeline(steps=num_steps)
                 transformers.append(('num', numeric_pipeline, numeric_features))

        # Categorical pipeline
        if categorical_features:
            cat_steps = []
            # Impute missing categoricals with a constant placeholder
            cat_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='_missing_')))
            if encode_categorical:
                 # Use OneHotEncoder, ignore unknown values encountered during transform
                 cat_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
            if cat_steps:
                 categorical_pipeline = Pipeline(steps=cat_steps)
                 transformers.append(('cat', categorical_pipeline, categorical_features))

        if not transformers:
            print("Warning: No numeric or categorical features found/selected for preprocessing.")
            return None # Or handle differently if needed

        try:
            preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough') # Keep other columns if any
            return preprocessor
        except Exception as e:
            print(f"Error building preprocessor: {e}")
            return None


    def run_modeling_task(self, data_source: str, request: str) -> str:
        """
        Performs the ML modeling task: load, preprocess, train, evaluate, save.

        Args:
            data_source: Path to the data file (e.g., 'data.csv') or CSV string.
            request: Natural language description of the modeling task
                     (e.g., "Predict salary based on experience and department",
                      "Build a model to classify employee department using salary and experience").

        Returns:
            A message indicating success (including evaluation metrics and model path) or failure.
        """
        print(f"\n--- Modeling Agent: Running Task ---")
        print(f"Data Source: {data_source}")
        print(f"Request: {request}")

        # 1. Parse Request
        params = self._parse_modeling_request(request)
        if not params or "error" in params:
            return f"Error: Could not parse modeling request. {params.get('error', '')}"

        target_col = params['target_column']
        task_type = params['task_type'].lower() # 'classification' or 'regression'
        output_base_filename = params.get('output_model_filename') # Base name for model/preprocessor files

        # 2. Load Data
        df = self._load_data(data_source)
        if df is None: return f"Error: Could not load data from '{data_source}'"
        if df.empty: return f"Error: Data loaded from '{data_source}' is empty."
        print(f"Modeling Agent: Data loaded. Shape: {df.shape}, Columns: {df.columns.tolist()}")

        # 3. Identify Features and Target
        if target_col not in df.columns: return f"Error: Target column '{target_col}' not found in data."

        feature_cols = params.get('feature_columns')
        if not feature_cols: # Use all columns except target if not specified
            feature_cols = [col for col in df.columns if col != target_col]
        else: # Validate specified feature columns
            missing_features = [col for col in feature_cols if col not in df.columns]
            if missing_features: return f"Error: Specified feature columns not found: {missing_features}"

        if not feature_cols: return "Error: No feature columns identified or specified."

        print(f"Modeling Agent: Target='{target_col}', Features={feature_cols}")
        X = df[feature_cols]
        y = df[target_col]

        # Basic check for target type vs task type
        if task_type == 'classification' and pd.api.types.is_numeric_dtype(y) and y.nunique() > 15: # Heuristic
             print(f"Warning: Task is classification, but target '{target_col}' looks numeric with many unique values. Check request parsing.")
        if task_type == 'regression' and not pd.api.types.is_numeric_dtype(y):
             print(f"Warning: Task is regression, but target '{target_col}' doesn't look numeric. Check request parsing.")
             # Attempt conversion? Or fail? For now, proceed but warn.
             try: y = pd.to_numeric(y)
             except: return f"Error: Regression task specified, but target column '{target_col}' could not be converted to numeric."


        # 4. Preprocessing & Data Splitting
        preprocessor = self._build_preprocessor(df, feature_cols, target_col, params)
        if preprocessor is None and (X.select_dtypes(include=np.number).empty or X.select_dtypes(include=['object','category']).empty):
             print("Modeling Agent: No preprocessing applied (likely only one data type or no steps enabled).")
             # Proceed without fitting preprocessor if it's effectively empty
             X_processed = X
        elif preprocessor:
             try:
                 print("Modeling Agent: Fitting preprocessor...")
                 # Fit on full X before splitting to avoid data leakage in fit,
                 # but transform train/test separately.
                 preprocessor.fit(X) # Fit the transformer
                 # We'll apply transform after splitting
             except Exception as e:
                 print(f"Error fitting preprocessor: {e}\n{traceback.format_exc()}")
                 return f"Error during data preprocessing fitting: {e}"
        else:
             # Error occurred during build_preprocessor
             return "Error: Failed to build data preprocessor."


        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Modeling Agent: Data split. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            # Apply preprocessing *after* splitting
            if preprocessor:
                 print("Modeling Agent: Transforming data with preprocessor...")
                 X_train_processed = preprocessor.transform(X_train)
                 X_test_processed = preprocessor.transform(X_test)
                 # Get feature names after transformation (important for some models/analysis)
                 try:
                      feature_names_out = preprocessor.get_feature_names_out()
                      print(f"Modeling Agent: Processed features ({len(feature_names_out)}): {feature_names_out[:10]}...") # Print first few
                 except Exception as name_err:
                      print(f"Warning: Could not get feature names from preprocessor: {name_err}")
                      feature_names_out = None
            else:
                 X_train_processed = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train # Ensure numpy array
                 X_test_processed = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
                 feature_names_out = feature_cols # Original features

        except Exception as e:
            print(f"Error during data splitting or preprocessing transform: {e}\n{traceback.format_exc()}")
            return f"Error during data splitting/transform: {e}"

        # 5. Model Selection & Training
        model_type_req = params.get('model_type')
        model = None
        print(f"Modeling Agent: Selecting model for task '{task_type}' (request: {model_type_req})...")

        try:
            if task_type == 'classification':
                if model_type_req == 'LogisticRegression': model = LogisticRegression(random_state=42, max_iter=1000)
                elif model_type_req == 'RandomForestClassifier': model = RandomForestClassifier(random_state=42)
                else: # Default classification
                    print("Using default classification model: RandomForestClassifier")
                    model = RandomForestClassifier(random_state=42)
            elif task_type == 'regression':
                if model_type_req == 'LinearRegression': model = LinearRegression()
                elif model_type_req == 'RandomForestRegressor': model = RandomForestRegressor(random_state=42)
                else: # Default regression
                    print("Using default regression model: RandomForestRegressor")
                    model = RandomForestRegressor(random_state=42)
            else:
                return f"Error: Unknown task type '{task_type}' determined."

            print(f"Modeling Agent: Training {model.__class__.__name__}...")
            model.fit(X_train_processed, y_train)
            print("Modeling Agent: Training complete.")

        except Exception as e:
            print(f"Error during model selection or training: {e}\n{traceback.format_exc()}")
            return f"Error during model training: {e}"

        # 6. Evaluation
        results = {}
        print("Modeling Agent: Evaluating model...")
        try:
            y_pred = model.predict(X_test_processed)
            default_metrics = params.get('evaluation_metrics') # Use requested or defaults

            if task_type == 'classification':
                metrics_to_calc = default_metrics or ['accuracy', 'f1-macro', 'report']
                if 'accuracy' in metrics_to_calc: results['accuracy'] = accuracy_score(y_test, y_pred)
                if 'report' in metrics_to_calc:
                    # Generate report, handle potential zero division issues
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    results['classification_report (dict)'] = report
                    # Add macro avg f1 if requested separately
                    if 'f1-macro' in metrics_to_calc: results['f1_macro_avg'] = report['macro avg']['f1-score']
                elif 'f1-macro' in metrics_to_calc: # Calculate if report wasn't generated but f1 was asked for
                     report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                     results['f1_macro_avg'] = report['macro avg']['f1-score']

            elif task_type == 'regression':
                metrics_to_calc = default_metrics or ['rmse', 'r2']
                if 'mse' in metrics_to_calc or 'rmse' in metrics_to_calc:
                    mse = mean_squared_error(y_test, y_pred)
                    results['MSE'] = mse
                    if 'rmse' in metrics_to_calc: results['RMSE'] = np.sqrt(mse)
                if 'r2' in metrics_to_calc: results['R2_score'] = r2_score(y_test, y_pred)

            print(f"Modeling Agent: Evaluation Metrics: {results}")

        except Exception as e:
            print(f"Error during model evaluation: {e}\n{traceback.format_exc()}")
            return f"Model trained, but error during evaluation: {e}"

        # 7. Save Model & Preprocessor (Optional)
        saved_paths = {}
        if output_base_filename:
            try:
                model_filename = f"{output_base_filename}_model.joblib"
                model_path = os.path.join(self.workspace_dir, model_filename)
                # Save model and preprocessor together for consistent application
                save_payload = {'model': model, 'preprocessor': preprocessor, 'feature_columns': feature_cols, 'target_column': target_col}
                joblib.dump(save_payload, model_path)
                print(f"Modeling Agent: Model and preprocessor saved to {model_path}")
                saved_paths['model_preprocessor_path'] = model_path
            except Exception as e:
                print(f"Error saving model/preprocessor: {e}")
                results['saving_error'] = str(e) # Add error to results

        # 8. Format and Return Result Message
        result_message = f"Successfully completed modeling task '{request}'.\n"
        result_message += "Evaluation Metrics:\n"
        for metric, value in results.items():
            if isinstance(value, float): result_message += f"- {metric}: {value:.4f}\n"
            elif metric == 'classification_report (dict)':
                 # Optionally format the dict report better here if needed
                 result_message += f"- Classification Report (Macro Avg F1): {value.get('macro avg', {}).get('f1-score', 'N/A'):.4f}\n"
            else: result_message += f"- {metric}: {value}\n" # Handle strings or other types

        if saved_paths:
            result_message += "\nSaved Artifacts:\n"
            for name, path in saved_paths.items():
                result_message += f"- {name}: {path}\n"

        return result_message


# --- Forecasting Agent Logic (NEW) ---
class ForecastingAgent:
    """
    Handles Time Series Forecasting tasks using SARIMAX.
    """
    def __init__(self, llm, workspace_dir: str):
        self.llm = llm
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
        print(f"Forecasting Agent using workspace: {self.workspace_dir}")

    def _parse_forecasting_request(self, request: str) -> Optional[Dict[str, Any]]:
        """Uses LLM to parse natural language request into forecasting parameters."""
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
            print(f"Forecasting Agent: Parsed Params: {parsed_params}")
            if not isinstance(parsed_params, dict): return {"error": "LLM parsing failed (not dict)."}
            if "error" in parsed_params: return parsed_params
            if not all(k in parsed_params for k in ['data_source_time_column', 'target_column', 'forecast_horizon']):
                 return {"error": "Missing required info: time_column, target_column, or forecast_horizon."}
            if not isinstance(parsed_params['forecast_horizon'], int) or parsed_params['forecast_horizon'] <= 0:
                 return {"error": "forecast_horizon must be a positive integer."}
            # Basic validation for order/seasonal_order if provided
            if parsed_params.get('order') and (not isinstance(parsed_params['order'], list) or len(parsed_params['order']) != 3):
                 return {"error": "Invalid 'order' format. Must be a list of 3 integers (p, d, q)."}
            if parsed_params.get('seasonal_order') and (not isinstance(parsed_params['seasonal_order'], list) or len(parsed_params['seasonal_order']) != 4):
                 return {"error": "Invalid 'seasonal_order' format. Must be a list of 4 integers (P, D, Q, s)."}

            return parsed_params
        except Exception as e:
            print(f"Error parsing forecasting request with LLM: {e}\n{traceback.format_exc()}")
            return {"error": f"LLM parsing failed: {e}"}

    def _load_ts_data(self, data_source: str, time_col: str, value_col: str, freq: Optional[str], datetime_format: Optional[str] = None) -> Optional[pd.Series]:
        """Loads time series data, ensuring DatetimeIndex and frequency."""
        path = data_source
        if not os.path.isabs(path): path = os.path.join(self.workspace_dir, data_source)

        df = None
        try:
            if os.path.exists(path) and path.lower().endswith('.csv'):
                print(f"Loading TS data from CSV: {path}")
                df = pd.read_csv(path)
            elif os.path.exists(path) and path.lower().endswith(('.xls', '.xlsx')):
                 print(f"Loading TS data from Excel: {path}")
                 df = pd.read_excel(path)
            elif ',' in data_source and '\n' in data_source: # Try string
                print("Attempting to load TS data from string (assuming CSV format).")
                df = pd.read_csv(io.StringIO(data_source))
            else:
                print(f"Forecasting Agent Load Error: Cannot read source '{data_source}'")
                return None

            if time_col not in df.columns: raise ValueError(f"Time column '{time_col}' not found.")
            if value_col not in df.columns: raise ValueError(f"Value column '{value_col}' not found.")

            # Convert to datetime and set index
            if datetime_format: df[time_col] = pd.to_datetime(df[time_col], format=datetime_format)
            else: df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
            df = df[[value_col]] # Keep only the target column
            df = df.sort_index()

            # Handle Frequency
            original_freq = pd.infer_freq(df.index)
            if freq:
                 print(f"Using specified frequency: {freq}")
                 df = df.asfreq(freq)
            elif original_freq:
                 freq = original_freq
                 df = df.asfreq(freq) # Ensure it's set
                 print(f"Inferred frequency: {freq}")
            else:
                 # Attempt to make regular if possible, otherwise raise error
                 median_diff = df.index.to_series().diff().median()
                 if pd.notna(median_diff):
                      inferred_freq_code = pd.tseries.frequencies.to_offset(median_diff)
                      if inferred_freq_code:
                           freq = inferred_freq_code.freqstr
                           print(f"Warning: Frequency not specified or directly inferable. Attempting to use median difference: {freq}")
                           df = df.asfreq(freq) # Try setting it
                      else:
                           raise ValueError("Could not infer frequency and no frequency specified. Cannot proceed with SARIMAX.")
                 else:
                      raise ValueError("Could not infer frequency and no frequency specified. Cannot proceed with SARIMAX.")

            # Handle missing values (simple forward fill for forecasting)
            initial_nan = df[value_col].isna().sum()
            if initial_nan > 0:
                df[value_col] = df[value_col].ffill().bfill() # Fill forward then backward
                print(f"Filled {initial_nan} missing values using ffill/bfill.")

            if df[value_col].isna().any():
                 raise ValueError("Data still contains NaNs after filling. Cannot proceed.")

            # Return just the Series, indexed by time, with frequency
            ts_series = df[value_col]
            ts_series.index.freq = freq # Ensure frequency is attached to the Series index
            return ts_series

        except Exception as e:
            print(f"Error loading or processing time series data from '{data_source}': {e}\n{traceback.format_exc()}")
            return None

    def _get_seasonal_period(self, freq: str) -> int:
        """Infer seasonal period 'm' from frequency string."""
        freq = freq.upper()
        if freq.startswith('D'): return 7
        if freq.startswith('W'): return 52 # Approximation
        if freq.startswith('M'): return 12
        if freq.startswith('Q'): return 4
        if freq.startswith('A') or freq.startswith('Y'): return 1 # Yearly
        if freq.startswith('H'): return 24
        if freq.startswith('T') or freq.startswith('MIN'): return 60
        if freq.startswith('S'): return 60
        print(f"Warning: Could not determine seasonal period for freq '{freq}'. Defaulting to 1.")
        return 1 # Default if unknown

    def _train_and_forecast(self, ts_series: pd.Series, params: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[Any], Optional[Dict]]:
        """Trains SARIMAX model and generates forecast."""
        horizon = params['forecast_horizon']
        order = params.get('order') or (1, 1, 1) # Default ARIMA order
        seasonal_order_req = params.get('seasonal_order')
        freq = ts_series.index.freqstr # Get freq from the series index

        if not freq:
             print("Error: Time series frequency is missing. Cannot determine seasonal period.")
             return None, None, None

        m = self._get_seasonal_period(freq)

        # Default seasonal order if not provided
        seasonal_order = seasonal_order_req or (1, 1, 1, m)
        # Ensure 's' in provided seasonal_order matches inferred 'm' if possible? Or trust user? Trust user for now.
        if seasonal_order_req and seasonal_order_req[3] != m:
             print(f"Warning: Provided seasonal period s={seasonal_order_req[3]} differs from inferred period m={m} based on frequency '{freq}'. Using provided s.")
             m = seasonal_order_req[3] # Use the user-provided seasonal period

        print(f"Forecasting Agent: Using SARIMAX order={order}, seasonal_order={seasonal_order}")

        # --- Evaluation Step (Train/Test Split) ---
        eval_metrics = {}
        test_size_heuristic = max(horizon, m, 10) # Use a test set at least as long as horizon or seasonality
        if len(ts_series) > 2 * test_size_heuristic: # Only evaluate if enough data
            print(f"Forecasting Agent: Performing evaluation on hold-out set (size={test_size_heuristic})...")
            train_ts = ts_series[:-test_size_heuristic]
            test_ts = ts_series[-test_size_heuristic:]
            try:
                eval_model = SARIMAX(train_ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                eval_model_fit = eval_model.fit(disp=False) # Suppress convergence output
                eval_forecast_obj = eval_model_fit.get_forecast(steps=len(test_ts))
                eval_preds = eval_forecast_obj.predicted_mean
                # Use the imported or fallback metric calculator
                eval_metrics = calculate_forecast_metrics(test_ts, eval_preds)
                print(f"Forecasting Agent: Evaluation Metrics: {eval_metrics}")
            except Exception as e:
                print(f"Warning: Error during evaluation step: {e}. Skipping evaluation.")
                eval_metrics = {"error": f"Evaluation failed: {e}"}
        else:
            print("Forecasting Agent: Not enough data for robust evaluation split. Skipping evaluation.")
            eval_metrics = {"info": "Skipped evaluation due to insufficient data."}


        # --- Final Model Training and Forecasting ---
        print(f"Forecasting Agent: Training final model on full data (size={len(ts_series)})...")
        try:
            model = SARIMAX(ts_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False) # Suppress convergence output during final fit
            print("Forecasting Agent: Final model training complete.")

            # Generate forecast
            forecast_obj = model_fit.get_forecast(steps=horizon)
            forecast_df = forecast_obj.summary_frame(alpha=0.05) # Get mean forecast and 95% CI
            forecast_df = forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']] # Select relevant columns
            forecast_df.rename(columns={'mean': 'predicted_value', 'mean_ci_lower': 'lower_ci', 'mean_ci_upper': 'upper_ci'}, inplace=True)

            print(f"Forecasting Agent: Forecast generated for {horizon} steps.")
            return forecast_df, model_fit, eval_metrics

        except Exception as e:
            print(f"Error during final model training or forecasting: {e}\n{traceback.format_exc()}")
            return None, None, eval_metrics # Return eval metrics even if final forecast fails


    def run_forecasting_task(self, data_source: str, request: str) -> str:
        """
        Performs the Time Series Forecasting task.

        Args:
            data_source: Path to the data file (e.g., 'calls.csv') or CSV string.
            request: Natural language description of the forecasting task.

        Returns:
            A message indicating success (including forecast summary, eval metrics, paths) or failure.
        """
        print(f"\n--- Forecasting Agent: Running Task ---")
        print(f"Data Source: {data_source}")
        print(f"Request: {request}")

        # 1. Parse Request
        params = self._parse_forecasting_request(request)
        if not params or "error" in params:
            return f"Error: Could not parse forecasting request. {params.get('error', '')}"

        time_col = params['data_source_time_column']
        target_col = params['target_column']
        freq = params.get('freq') # Can be None initially
        output_forecast_filename = params.get('output_forecast_filename')
        output_model_filename = params.get('output_model_filename')

        # 2. Load Data
        ts_series = self._load_ts_data(data_source, time_col, target_col, freq)
        if ts_series is None:
            return f"Error: Could not load or process time series data from '{data_source}'."
        if ts_series.empty:
             return f"Error: Time series data loaded from '{data_source}' is empty or invalid."

        print(f"Forecasting Agent: Time series data loaded. Length: {len(ts_series)}, Freq: {ts_series.index.freqstr}")

        # 3. Train and Forecast (includes evaluation)
        forecast_df, fitted_model, eval_metrics = self._train_and_forecast(ts_series, params)

        if forecast_df is None or fitted_model is None:
            eval_info = f"\nEvaluation Metrics (if attempted): {eval_metrics}" if eval_metrics else ""
            return f"Error: Failed to train model or generate forecast.{eval_info}"

        # 4. Save Results (Optional)
        saved_paths = {}
        if output_forecast_filename:
            try:
                forecast_filename_csv = f"{output_forecast_filename}_forecast.csv"
                forecast_path = os.path.join(self.workspace_dir, forecast_filename_csv)
                forecast_df.to_csv(forecast_path)
                print(f"Forecasting Agent: Forecast data saved to {forecast_path}")
                saved_paths['forecast_data_path'] = forecast_path
            except Exception as e:
                print(f"Error saving forecast data: {e}")
                eval_metrics['forecast_saving_error'] = str(e)

        if output_model_filename:
            try:
                model_filename_joblib = f"{output_model_filename}_model.joblib"
                model_path = os.path.join(self.workspace_dir, model_filename_joblib)
                # Save the fitted SARIMAX results object
                joblib.dump(fitted_model, model_path)
                print(f"Forecasting Agent: Trained model saved to {model_path}")
                saved_paths['trained_model_path'] = model_path
            except Exception as e:
                print(f"Error saving trained model: {e}")
                eval_metrics['model_saving_error'] = str(e)

        # 5. Format and Return Result Message
        result_message = f"Successfully completed forecasting task '{request}'.\n"
        result_message += f"Forecast Horizon: {params['forecast_horizon']} steps.\n"

        result_message += "\nEvaluation Metrics (on hold-out set, if performed):\n"
        if eval_metrics:
            for metric, value in eval_metrics.items():
                 if isinstance(value, float): result_message += f"- {metric}: {value:.4f}\n"
                 else: result_message += f"- {metric}: {value}\n"
        else:
             result_message += "- Evaluation not performed.\n"

        result_message += "\nForecast Summary (first 5 steps):\n"
        result_message += forecast_df.head().to_string() + "\n"

        if saved_paths:
            result_message += "\nSaved Artifacts:\n"
            for name, path in saved_paths.items():
                result_message += f"- {name}: {path}\n"

        return result_message


# --- Supervisor Agent Class ---
class SupervisorAgent:
    def __init__(self, model_name: str = "mistral", temperature: float = 0.7, db_uri: Optional[str] = None):
        self.model_name = model_name; self.temperature = temperature; self.db_uri = db_uri
        self.llm = ChatOllama(model=self.model_name, temperature=self.temperature)
        self.workspace_dir = os.path.abspath("./agent_workspace"); os.makedirs(self.workspace_dir, exist_ok=True)
        print(f"Supervisor using workspace: {self.workspace_dir}")
        # Initialize specialized agents/logic providers
        self.sql_agent_instance = self._initialize_sql_agent()
        self.visualization_agent_instance = self._initialize_visualization_agent()
        self.modeling_agent_instance = self._initialize_modeling_agent()
        self.forecasting_agent_instance = self._initialize_forecasting_agent() # NEW
        self.tools = self._get_tools()
        self.agent_executor = self._create_agent_executor()

    def _initialize_sql_agent(self) -> Optional[SQLAgent]:
        if not self.db_uri: return None
        try: return SQLAgent(llm=self.llm, db_uri=self.db_uri, verbose=True)
        except Exception as e: print(f"Warn: SQL Agent init fail: {e}"); return None
    def _initialize_visualization_agent(self) -> Optional[VisualizationAgent]:
        try: return VisualizationAgent(llm=self.llm, workspace_dir=self.workspace_dir)
        except Exception as e: print(f"Warn: Viz Agent init fail: {e}"); return None
    def _initialize_modeling_agent(self) -> Optional[ModelingAgent]:
         try: return ModelingAgent(llm=self.llm, workspace_dir=self.workspace_dir)
         except Exception as e: print(f"Warn: Modeling Agent init fail: {e}"); return None
    # NEW: Initialize Forecasting Agent Logic
    def _initialize_forecasting_agent(self) -> Optional[ForecastingAgent]:
         try:
             print("Initializing Forecasting Agent Logic...")
             return ForecastingAgent(llm=self.llm, workspace_dir=self.workspace_dir)
         except Exception as e:
              print(f"Warning: Failed to initialize Forecasting Agent: {e}. Forecasting tool will not be available.")
              return None

    # --- Tool Definition Wrapper for Forecasting ---
    def _run_forecasting_tool(self, data_source: str, request: str) -> str:
        """
        Tool for the Supervisor Agent to run time series forecasting tasks.
        Takes data (file path or CSV string) and a natural language request describing the
        forecasting task (e.g., "Forecast sales for the next 12 months using sales_data.csv").
        It handles loading, preprocessing, training (SARIMAX), forecasting, evaluation, and optional saving.

        Args:
            data_source: Path to the time series data file (e.g., 'sales.csv' in workspace) or raw CSV data as a string.
            request: Natural language description of the forecasting task. CRITICAL: Must mention the target column to forecast, the time/date column in the source data, and the desired forecast horizon (number of steps). Optionally specify frequency ('D', 'MS', etc.) or model parameters.
        """
        if not self.forecasting_agent_instance:
            return "Forecasting Agent is not configured or failed to initialize."
        try:
            result_message = self.forecasting_agent_instance.run_forecasting_task(
                data_source=data_source,
                request=request
            )
            return result_message
        except Exception as e:
            print(f"Unexpected error in forecasting tool wrapper: {e}\n{traceback.format_exc()}")
            return f"An unexpected error occurred while running the forecasting tool: {e}"

    # --- Tool Definition Wrappers for Modeling, Viz & SQL ---
    def _run_modeling_tool(self, data_source: str, request: str) -> str:
        if not self.modeling_agent_instance: return "Modeling Agent not init."
        try: return self.modeling_agent_instance.run_modeling_task(data_source, request)
        except Exception as e: print(f"Modeling tool error: {e}\n{traceback.format_exc()}"); return f"Modeling tool error: {e}"
    def _run_visualization_tool(self, data_source: str, request: str, output_filename: str) -> str:
        if not self.visualization_agent_instance: return "Viz Agent not init."
        try: return self.visualization_agent_instance.generate_plot(data_source, request, output_filename)
        except Exception as e: return f"Viz tool error: {e}"
    def _run_sql_query_tool(self, natural_language_query: str) -> str:
        if not self.sql_agent_instance: return "SQL Agent not init."
        try:
            response = self.sql_agent_instance.run(natural_language_query)
            if response["error"]: return f"SQL Error: {response['error']}\nSQL: {response['generated_sql']}"
            return f"Result:\n{response['result']}\n\nSQL:\n```sql\n{response['generated_sql']}\n```"
        except Exception as e: return f"SQL tool error: {e}"


    def _get_tools(self) -> List[BaseTool]:
        """Gets all tools available to the Supervisor Agent."""
        all_tools = []
        # File Management Tools
        try:
            fm_toolkit = FileManagementToolkit(root_dir=self.workspace_dir); all_tools.extend(fm_toolkit.get_tools())
            print(f"Loaded File Tools: {[t.name for t in fm_toolkit.get_tools()]}")
        except Exception as e: print(f"Warn: File tools init fail: {e}")
        # SQL Tool
        if self.sql_agent_instance:
            sql_tool = tool(name="query_sql_database", func=self._run_sql_query_tool, description="Query SQL DB with natural language."); all_tools.append(sql_tool)
            print(f"Loaded SQL Tool: {sql_tool.name}")
        # Visualization Tool
        if self.visualization_agent_instance:
             viz_tool = tool(name="create_visualization", func=self._run_visualization_tool, description=f"Create plot from data (path/CSV string in '{self.workspace_dir}'), save as image. Args: data_source, request, output_filename."); all_tools.append(viz_tool)
             print(f"Loaded Viz Tool: {viz_tool.name}")
        # Modeling Tool
        if self.modeling_agent_instance:
             modeling_tool = tool(name="run_modeling_task", func=self._run_modeling_tool, description=f"Train/evaluate ML model (classification/regression) from data (path/CSV string in '{self.workspace_dir}'). Request MUST specify target column. Returns metrics & optional saved model path. Args: data_source, request.")
             all_tools.append(modeling_tool)
             print(f"Loaded Modeling Tool: {modeling_tool.name}")
        # Forecasting Tool (NEW)
        if self.forecasting_agent_instance:
             forecasting_tool = tool(
                 name="run_time_series_forecast",
                 func=self._run_forecasting_tool,
                 description=f"""Use this tool to generate a time series forecast using the SARIMAX model.
You MUST provide:
1. `data_source`: Path to the time series data file (e.g., 'sales_data.csv' located in '{self.workspace_dir}') OR the actual data as a CSV-formatted string.
2. `request`: A natural language description of the forecasting task. CRITICAL: Clearly state the column containing the values to forecast (target), the column containing the date/time information, and the desired forecast horizon (number of steps/periods). Optionally specify the data frequency (e.g., 'D' for daily, 'MS' for monthly start) if known, or specific SARIMAX orders. Example: "Forecast the 'call_volume' for the next 7 days using 'call_data.csv'. The time column is 'timestamp' and frequency is daily ('D'). Save the forecast."
The tool handles loading, preprocessing, training, forecasting, evaluation, and optional saving. It returns evaluation metrics and paths to saved forecast/model files.""",
             )
             all_tools.append(forecasting_tool)
             print(f"Loaded Forecasting Tool: {forecasting_tool.name}")

        if not all_tools: print("Warning: No tools loaded for Supervisor.")
        return all_tools

    def _create_agent_executor(self) -> AgentExecutor:
        agent_type = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
        print(f"Creating Supervisor Agent Executor with tools: {[t.name for t in self.tools]}")
        if not self.tools: raise ValueError("Cannot create agent executor with no tools.")
        try:
            # Increased max_iterations slightly for potentially longer forecasting tasks
            return initialize_agent(tools=self.tools, llm=self.llm, agent=agent_type, verbose=True,
                                    handle_parsing_errors="Check your output format!", max_iterations=25,
                                    early_stopping_method="generate")
        except Exception as e: print(f"Error initializing supervisor agent executor: {e}"); raise

    def run(self, user_input: str) -> Dict[str, Any]:
        print(f"\n--- Running Supervisor Agent ---"); print(f"Input: {user_input}")
        if not self.agent_executor: return {"error": "Supervisor agent executor not initialized."}
        try:
            result = self.agent_executor.invoke({"input": user_input})
            print(f"--- Supervisor Agent Finished ---"); return result
        except Exception as e:
            print(f"Error during Supervisor Agent execution: {e}\n{traceback.format_exc()}")
            output = None; result = locals().get('result')
            if result and isinstance(result, dict): output = result.get('output')
            return {"error": str(e), "partial_output": output}


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    OLLAMA_MODEL = "mistral"; DB_TYPE = "sqlite"; DB_NAME = "sample_company.db"
    DB_URI = f"sqlite:///{DB_NAME}" if DB_TYPE == "sqlite" else None

    # --- Database Setup (SQLite Example) ---
    if DB_URI and not os.path.exists(DB_NAME):
        print(f"Creating dummy SQLite database: {DB_NAME}"); import sqlite3
        conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
        cursor.execute('''CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary REAL, experience_years INTEGER)''')
        cursor.executemany("INSERT INTO employees (name, department, salary, experience_years) VALUES (?, ?, ?, ?)", [('Alice','Eng',90,5),('Bob','Sales',75,3),('Charlie','Eng',95,7),('David','Sales',80,4),('Eve','HR',65,2),('Frank','Eng',110,10),('Grace','HR',70,3)])
        conn.commit(); conn.close(); print("Dummy database created/populated.")
    elif DB_URI: print(f"Using existing database: {DB_NAME}")

    # --- Initialize Supervisor ---
    print("\n--- Initializing Supervisor Agent ---")
    supervisor = SupervisorAgent(model_name=OLLAMA_MODEL, temperature=0.1, db_uri=DB_URI)

    # --- Prepare Data Files (using Supervisor) ---
    DATA_FILE_FOR_MODELING = "employees_for_modeling.csv"
    TIME_SERIES_DATA_FILE = "sample_ts_data.csv"

    # Prepare modeling data
    if DB_URI:
        print(f"\n--- Preparing Data File: {DATA_FILE_FOR_MODELING} ---")
        prep_task_model = f"Query DB for department, salary, experience_years. Save to '{DATA_FILE_FOR_MODELING}'. Tell me when done."
        prep_result_model = supervisor.run(prep_task_model); print("\nData Prep (Model) Result:", prep_result_model)
        data_file_path_model = os.path.join(supervisor.workspace_dir, DATA_FILE_FOR_MODELING)
        if not os.path.exists(data_file_path_model): print(f"\nFATAL: Modeling data file not created."); exit()
        else: print(f"\nModeling data file created: {data_file_path_model}")
    else: data_file_path_model = None; print("\nSkipping modeling data prep (no DB_URI).")

    # Prepare time series data (Create synthetic data if DB not available or doesn't have time data)
    print(f"\n--- Preparing Data File: {TIME_SERIES_DATA_FILE} ---")
    ts_file_path = os.path.join(supervisor.workspace_dir, TIME_SERIES_DATA_FILE)
    if not os.path.exists(ts_file_path):
         print("Creating synthetic time series data...")
         dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
         # Simulate volume with trend and seasonality
         volume = (np.arange(100) * 0.5 + 50 +
                   np.sin(np.arange(100) * 2 * np.pi / 7) * 10 + # Weekly seasonality
                   np.random.normal(0, 5, 100)) # Noise
         ts_df = pd.DataFrame({'Date': dates, 'CallVolume': volume.astype(int)})
         ts_df.to_csv(ts_file_path, index=False)
         print(f"Synthetic time series data saved to {ts_file_path}")
    else:
         print(f"Using existing time series data file: {ts_file_path}")


    # --- Run Modeling Task ---
    if data_file_path_model:
        print("\n--- Task: Regression Modeling ---")
        task_model = f"Use '{DATA_FILE_FOR_MODELING}' to predict 'salary' from 'department', 'experience_years'. Save model as 'salary_predictor'."
        result_model = supervisor.run(task_model); print("\nModeling Task Result:", result_model)
    else: print("\nSkipping modeling task (no data file).")

    # --- Run Forecasting Task (NEW) ---
    if os.path.exists(ts_file_path):
        print("\n--- Task 7: Time Series Forecasting ---")
        task7 = f"""
        Using the data in '{TIME_SERIES_DATA_FILE}', generate a forecast for 'CallVolume'.
        The time column is 'Date'. The frequency is daily ('D').
        Forecast the volume for the next 14 days.
        Evaluate the model performance.
        Save the forecast results with the base name 'call_forecast'.
        Save the trained model with the base name 'call_volume_forecaster'.
        Report the evaluation metrics, forecast summary, and saved file paths.
        """
        result7 = supervisor.run(task7)
        print("\nTask 7 (Forecasting) Result:")
        print(result7)
        # Verify output files exist
        forecast_csv_path = os.path.join(supervisor.workspace_dir, "call_forecast_forecast.csv")
        model_joblib_path = os.path.join(supervisor.workspace_dir, "call_volume_forecaster_model.joblib")
        if os.path.exists(forecast_csv_path): print(f"\nSUCCESS: Forecast CSV found at: {forecast_csv_path}")
        else: print(f"\nWARNING: Forecast CSV NOT found at: {forecast_csv_path}")
        if os.path.exists(model_joblib_path): print(f"SUCCESS: Forecast model file found at: {model_joblib_path}")
        else: print(f"\nWARNING: Forecast model file NOT found at: {model_joblib_path}")

    else:
        print("\nSkipping forecasting task because time series data file was not prepared.")

    print("\n--- Agent Execution Finished ---")