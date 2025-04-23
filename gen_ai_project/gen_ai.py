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

# --- RAG & Memory Components ---
import chromadb
from chromadb.utils import embedding_functions
from mem0 import Memory
# --- Document Processing ('docling' stand-in) ---
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument # Rename to avoid conflict

# --- Agent State (Example for potential graph expansion) ---
import operator
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], operator.add]
#     # Add other state elements if building a more complex graph

# --- Configuration (Add RAG/Mem0 specific) ---
CHROMA_DB_PATH = "./chroma_db_supervisor" # Directory to store persistent ChromaDB data
COLLECTION_NAME = "supervisor_knowledge_base"
# Using default SentenceTransformer (runs locally, free)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# For OpenAI embeddings (requires API key and billing):
# EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
# embedding_func = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL_NAME)

# --- Helper Functions for Document Processing ('docling' stand-in) ---

def load_and_split_documents(source_path: str, workspace_dir: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[LangchainDocument]:
    """
    Loads documents from a file or directory (relative to workspace) and splits them.
    Represents the 'docling' functionality. Supports .txt, .pdf, .csv, .md etc. via UnstructuredFileLoader.
    """
    full_path = source_path
    if not os.path.isabs(source_path):
        full_path = os.path.join(workspace_dir, source_path)

    if not os.path.exists(full_path):
        print(f"Error: Source path not found: {full_path}")
        return []

    documents = []
    print(f"Loading documents from: {full_path}")
    if os.path.isdir(full_path):
        # Load all supported files from a directory
        # Consider adding specific loaders for better control if needed (e.g., PyPDFLoader)
        loader = DirectoryLoader(full_path, glob="**/*.*", loader_cls=UnstructuredFileLoader, show_progress=True, use_multithreading=True)
        documents = loader.load()
    elif os.path.isfile(full_path):
        # Use UnstructuredFileLoader for broad single-file support
        loader = UnstructuredFileLoader(full_path)
        documents = loader.load()
    else:
        print(f"Error: Path is neither a file nor a directory: {full_path}")
        return []

    if not documents:
        print("No documents loaded.")
        return []

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Loaded and split {len(documents)} document(s) into {len(split_docs)} chunks.")
    return split_docs



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


# --- Supervisor Agent Class (Modified for RAG and Autonomy) ---
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
        self.forecasting_agent_instance = self._initialize_forecasting_agent()

        # --- Initialize RAG Components ---
        self.embedding_func = self._initialize_embedding_function()
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_func # Use the initialized function
        )
        print(f"ChromaDB collection '{COLLECTION_NAME}' loaded/created at '{CHROMA_DB_PATH}'.")

        # --- Initialize Memory ---
        self.mem0_memory = Memory()
        print("Mem0 initialized.")

        # --- Initialize Tools & Agent Executor ---
        self.tools = self._get_tools() # Now includes RAG tool
        self.agent_executor = self._create_agent_executor()

    def _initialize_embedding_function(self):
        """Initializes the embedding function based on configuration."""
        # Add logic here if you want to switch between OpenAI and SentenceTransformers easily
        print(f"Initializing embedding function: {EMBEDDING_MODEL_NAME}")
        # Default to SentenceTransformer
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        # Example for OpenAI:
        # openai_key = os.getenv("OPENAI_API_KEY")
        # if not openai_key: raise ValueError("OPENAI_API_KEY needed for OpenAI embeddings")
        # return embedding_functions.OpenAIEmbeddingFunction(api_key=openai_key, model_name=EMBEDDING_MODEL_NAME)

    # --- Keep existing agent initializers ---
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
    def _initialize_forecasting_agent(self) -> Optional[ForecastingAgent]:
         try: return ForecastingAgent(llm=self.llm, workspace_dir=self.workspace_dir)
         except Exception as e: print(f"Warn: Forecast Agent init fail: {e}"); return None

    # --- RAG Helper Methods ---
    def _add_documents_to_chroma(self, documents: List[LangchainDocument]):
        """Adds processed document chunks to the ChromaDB collection."""
        if not documents: print("No documents provided to add."); return
        ids = [str(uuid.uuid4()) for _ in documents] # Use uuid for unique IDs
        contents = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        try:
            print(f"Adding {len(contents)} document chunks to ChromaDB '{COLLECTION_NAME}'...")
            self.collection.add(documents=contents, metadatas=metadatas, ids=ids)
            print(f"Successfully added {len(contents)} chunks.")
            # Optional: Add info about the ingestion to Mem0
            sources_str = ", ".join(list(set(m.get('source', 'unknown') for m in metadatas)))
            self.mem0_memory.add(f"Ingested {len(contents)} chunks from sources: {sources_str}", user_id="supervisor_system", agent_id="doc_ingestor")
        except Exception as e: print(f"Error adding documents to ChromaDB: {e}")

    def _retrieve_relevant_context(self, query: str, n_results: int = 5) -> List[str]:
        """Queries ChromaDB for relevant document chunks."""
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results, include=['documents'])
            if results and results.get('documents') and results['documents'][0]:
                print(f"Retrieved {len(results['documents'][0])} relevant context chunks from ChromaDB.")
                return results['documents'][0]
            else: print("No relevant documents found in ChromaDB."); return []
        except Exception as e: print(f"Error querying ChromaDB: {e}"); return []

    # --- Tool Definition Wrappers (Keep existing, add RAG tool) ---
    def _run_modeling_tool(self, data_source: str, request: str) -> str: # Keep as is
        if not self.modeling_agent_instance: return "Modeling Agent not init."
        try: return self.modeling_agent_instance.run_modeling_task(data_source, request)
        except Exception as e: print(f"Modeling tool error: {e}\n{traceback.format_exc()}"); return f"Modeling tool error: {e}"
    def _run_visualization_tool(self, data_source: str, request: str, output_filename: str) -> str: # Keep as is
        if not self.visualization_agent_instance: return "Viz Agent not init."
        try: return self.visualization_agent_instance.generate_plot(data_source, request, output_filename)
        except Exception as e: return f"Viz tool error: {e}"
    def _run_sql_query_tool(self, natural_language_query: str) -> str: # Keep as is
        if not self.sql_agent_instance: return "SQL Agent not init."
        try:
            response = self.sql_agent_instance.run(natural_language_query)
            if response["error"]: return f"SQL Error: {response['error']}\nSQL: {response['generated_sql']}"
            return f"Result:\n{response['result']}\n\nSQL:\n```sql\n{response['generated_sql']}\n```"
        except Exception as e: return f"SQL tool error: {e}"
    def _run_forecasting_tool(self, data_source: str, request: str) -> str: # Keep as is
        if not self.forecasting_agent_instance: return "Forecast Agent not init."
        try: return self.forecasting_agent_instance.run_forecasting_task(data_source, request)
        except Exception as e: print(f"Forecast tool error: {e}\n{traceback.format_exc()}"); return f"Forecast tool error: {e}"

    # --- NEW Tool for Adding Knowledge ---
    def _add_knowledge_tool(self, source_path: str) -> str:
        """
        Tool to add knowledge to the supervisor's knowledge base.
        Loads and processes documents from a given file path or directory path
        (relative to the agent's workspace) and stores them in the vector database.

        Args:
            source_path (str): The path to the document file (e.g., 'docs/policy.pdf')
                               or directory (e.g., 'knowledge_files/') within the workspace.
        """
        print(f"\n--- Knowledge Addition Tool ---")
        print(f"Source path: {source_path}")
        try:
            # Use the helper function for loading/splitting
            documents = load_and_split_documents(source_path, self.workspace_dir)
            if not documents:
                return f"Failed to load or split documents from '{source_path}'. No documents were added."

            # Use the helper function to add to Chroma
            self._add_documents_to_chroma(documents)
            return f"Successfully processed and added knowledge from '{source_path}' to the knowledge base."

        except Exception as e:
            print(f"Error in knowledge addition tool: {e}\n{traceback.format_exc()}")
            return f"An error occurred while adding knowledge from '{source_path}': {e}"


    def _get_tools(self) -> List[BaseTool]:
        """Gets all tools available to the Supervisor Agent, including RAG."""
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
        # Forecasting Tool
        if self.forecasting_agent_instance:
             forecasting_tool = tool(name="run_time_series_forecast", func=self._run_forecasting_tool, description=f"Generate time series forecast using SARIMAX. Args: data_source (path/CSV string in '{self.workspace_dir}'), request (natural language, MUST specify target col, time col, horizon; optionally freq, orders). Returns metrics, forecast summary, optional saved paths.")
             all_tools.append(forecasting_tool)
             print(f"Loaded Forecasting Tool: {forecasting_tool.name}")

        # --- NEW Knowledge Addition Tool ---
        knowledge_tool = tool(
            name="add_knowledge_base_document",
            func=self._add_knowledge_tool,
            description=f"Add knowledge from a document or directory to the assistant's knowledge base for later retrieval. Use this to teach the assistant about company policies, project details, etc. Args: source_path (path to file/dir in workspace '{self.workspace_dir}')."
        )
        all_tools.append(knowledge_tool)
        print(f"Loaded Knowledge Tool: {knowledge_tool.name}")

        if not all_tools: print("Warning: No tools loaded for Supervisor.")
        return all_tools

    def _create_agent_executor(self) -> AgentExecutor:
        # Use a slightly more robust agent type if available, or stick with the original
        # agent_type = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
        agent_type = AgentType.OPENAI_FUNCTIONS # Often works well with Ollama models supporting function calling
        # agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION # A fallback if others fail

        print(f"Creating Supervisor Agent Executor ({agent_type}) with tools: {[t.name for t in self.tools]}")
        if not self.tools: raise ValueError("Cannot create agent executor with no tools.")

        # Define the core prompt, including instructions about using the knowledge base implicitly
        # Note: The agent might not *explicitly* call a "retrieve" tool, but the final synthesis step will use it.
        system_prompt = """You are a helpful and comprehensive assistant for company employees.
You have access to several tools to perform tasks like querying databases, creating visualizations, training models, forecasting time series, managing files, and adding knowledge.
You also have access to an internal knowledge base and conversation memory to provide more accurate and context-aware answers.
When asked a question, first try to use your specialized tools if the request clearly matches their capabilities (SQL, plotting, modeling, forecasting, file management).
If the question is general, informational, or asks about previously discussed topics or ingested knowledge, rely on your internal knowledge and memory.
Be clear and concise in your responses. If you use a tool, summarize the result. If you use the knowledge base or memory, indicate that.
If you cannot answer a question or perform a task, state that clearly.
Okay, begin!"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history", optional=True), # For potential future history management within agent
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        try:
            # Use the prompt with the agent initialization if supported by the type
            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=agent_type,
                verbose=True,
                handle_parsing_errors="Check your output format and try again!", # More helpful error
                max_iterations=15, # Keep reasonable default
                early_stopping_method="generate",
                # agent_kwargs={"system_message": system_prompt} # Pass system message if agent_type supports it
                # prompt=prompt # Pass full prompt if agent_type supports it
            )
            # If the agent type doesn't directly support prompt, the system message might need to be part of the input or handled differently.
            # For now, we rely on the final synthesis step for RAG integration.
            return agent # initialize_agent returns the AgentExecutor directly
        except Exception as e:
            print(f"Error initializing supervisor agent executor: {e}\n{traceback.format_exc()}")
            raise

    # --- METHOD FOR INTERACTIVE CHAT ---
    def run(self, user_input: str, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Runs the agent executor for interactive chat and enhances the response with RAG and Memory.
        This is the primary method for user interaction.
        """
        print(f"\n--- Running Supervisor Agent (Interactive) ---"); print(f"User ({user_id}) Input: {user_input}")
        if not self.agent_executor: return {"error": "Supervisor agent executor not initialized."}

        agent_response = {}
        final_answer = "An error occurred."
        try:
            # 1. Run the main agent executor (might use tools)
            # Pass input in the expected format (often a dictionary)
            agent_response = self.agent_executor.invoke({"input": user_input})
            initial_output = agent_response.get("output", "Agent did not produce a direct output.")
            print(f"Initial Agent Output: {initial_output}")

            # 2. Retrieve context from ChromaDB (RAG)
            retrieved_context_chunks = self._retrieve_relevant_context(user_input, n_results=3) # Get top 3 chunks
            retrieved_context = "\n\n".join(retrieved_context_chunks) if retrieved_context_chunks else "No relevant context found in knowledge base."

            # 3. Retrieve relevant memories from Mem0
            mem0_context = ""
            try:
                relevant_memories = self.mem0_memory.search(query=user_input, user_id=user_id, limit=5)
                if relevant_memories:
                    mem0_context = "\n\nRelevant past interactions:\n" + "\n".join([f"- {m['text']}" for m in relevant_memories])
                    print(f"Retrieved {len(relevant_memories)} relevant memories from Mem0.")
                else:
                    print("No relevant memories found in Mem0.")
            except Exception as e:
                print(f"Could not retrieve memories from Mem0: {e}")


            # 4. Synthesize Final Answer using LLM with all context
            synthesis_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=f"""You are synthesizing the final answer for a user based on their query, the initial attempt by an assistant, retrieved knowledge base context, and conversation history.
Your goal is to provide the most accurate, complete, and helpful response.
- Prioritize information from the 'Retrieved Context' if it directly answers the query.
- Use the 'Initial Assistant Output' if it involved a tool execution (like SQL query, file operation, model training) and summarize its results clearly.
- Incorporate 'Relevant Past Interactions' for conversational continuity.
- If the initial output seems sufficient and context/memory don't add much, you can rely primarily on the initial output.
- If the initial output failed or was irrelevant, construct the answer primarily from the retrieved context and memory.
- Be concise and directly answer the user's query.
- Do not mention the synthesis process itself, just provide the final answer.
"""),
                HumanMessage(content=f"""Okay, synthesize the final response based on the following:

User Query:
{user_input}

Initial Assistant Output:
{initial_output}

Retrieved Context from Knowledge Base:
{retrieved_context}

Relevant Past Interactions:
{mem0_context}

Synthesized Final Answer:
""")
            ])

            synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
            print("\n--- Synthesizing Final Answer ---")
            final_answer = synthesis_chain.invoke({})
            print(f"Synthesized Answer: {final_answer}")

            # 5. Add final interaction to Mem0
            try:
                self.mem0_memory.add(text=f"User: {user_input}\nAssistant: {final_answer}", user_id=user_id, agent_id="supervisor_final")
                print("Saved final interaction to Mem0.")
            except Exception as e:
                print(f"Could not save final interaction to Mem0: {e}")

            # Add the synthesized answer to the original response dict
            agent_response['final_output'] = final_answer

        except Exception as e:
            print(f"Error during Supervisor Agent execution or RAG synthesis: {e}\n{traceback.format_exc()}")
            final_answer = f"An error occurred: {e}"
            agent_response['error'] = str(e)
            agent_response['final_output'] = final_answer # Ensure final_output exists even on error

        print(f"--- Supervisor Agent Finished (Interactive) ---")
        return agent_response

    # --- METHOD FOR AUTONOMOUS EXECUTION ---
    def _execute_autonomous_task(self, task_prompt: str, task_id: str = "autonomous_task") -> Dict[str, Any]:
        """
        Executes a task using the agent executor, bypassing the usual RAG/Memory synthesis.
        Logs the output or saves it as needed. Designed for scheduled/automated execution.

        Args:
            task_prompt (str): The prompt describing the task for the agent.
            task_id (str): A unique identifier for this task run (used for logging).

        Returns:
            Dict[str, Any]: The raw response dictionary from the agent executor, or an error dict.
        """
        print(f"\n--- Running Autonomous Task ({task_id}) ---")
        print(f"Task Prompt: {task_prompt}")
        if not self.agent_executor:
            print("Error: Supervisor agent executor not initialized.")
            return {"error": "Supervisor agent executor not initialized."}

        try:
            # Invoke the agent with the predefined task prompt
            # Note: This bypasses the RAG/Memory synthesis done in the `run` method.
            # The agent will still use its tools as needed based on the prompt.
            response = self.agent_executor.invoke({"input": task_prompt})
            output = response.get("output", "Agent did not produce output for autonomous task.")
            print(f"Autonomous Task Output: {output}")

            # --- Handle the output ---
            # You might want to:
            # 1. Log the output to a file.
            # 2. Save results if the task involved file generation (the tool should handle this).
            # 3. Send a notification (requires adding notification capabilities).
            # 4. Update a status database.

            # Example: Log to a dedicated file in the workspace
            log_file_path = os.path.join(self.workspace_dir, f"{task_id}_log.txt")
            with open(log_file_path, "a") as log_file:
                timestamp = datetime.now().isoformat()
                log_file.write(f"--- {timestamp} ---\n")
                log_file.write(f"Task: {task_prompt}\n")
                log_file.write(f"Output:\n{output}\n\n")
            print(f"Autonomous task output logged to: {log_file_path}")

            # Add interaction to Mem0 if desired (maybe with a system user_id)
            try:
                self.mem0_memory.add(
                    text=f"Autonomous Task: {task_prompt}\nResult: {output}",
                    user_id="system_scheduler", # Identify scheduled tasks
                    agent_id=task_id
                )
                print("Saved autonomous task interaction to Mem0.")
            except Exception as e:
                print(f"Could not save autonomous task interaction to Mem0: {e}")


            return response # Return the raw response dictionary

        except Exception as e:
            print(f"Error during autonomous task execution ({task_id}): {e}\n{traceback.format_exc()}")
            # Log the error as well
            error_log_file_path = os.path.join(self.workspace_dir, f"{task_id}_error_log.txt")
            with open(error_log_file_path, "a") as log_file:
                 timestamp = datetime.now().isoformat()
                 log_file.write(f"--- {timestamp} ---\n")
                 log_file.write(f"Task: {task_prompt}\n")
                 log_file.write(f"Error: {e}\n{traceback.format_exc()}\n\n")
            print(f"Autonomous task error logged to: {error_log_file_path}")
            return {"error": str(e)}

    # --- Potential specific autonomous methods (Examples) ---
    # def perform_daily_report(self):
    #     """Example specific autonomous task method."""
    #     task_prompt = "Generate the daily sales summary report using the SQL database and save it to 'daily_sales.txt'."
    #     self._execute_autonomous_task(task_prompt, task_id="daily_sales_report")
    #
    # def perform_weekly_forecast(self):
    #     """Example specific autonomous task method."""
    #     task_prompt = f"Run a 7-day forecast for 'CallVolume' using data from 'call_data.csv' (time col 'Timestamp', freq 'H') and save the model as 'call_forecast_model'."
    #     self._execute_autonomous_task(task_prompt, task_id="weekly_call_forecast")


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    OLLAMA_MODEL = "mistral"; DB_TYPE = "sqlite"; DB_NAME = "sample_company.db"
    DB_URI = f"sqlite:///{DB_NAME}" if DB_TYPE == "sqlite" else None

    # --- Database Setup (SQLite Example - Keep as is) ---
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

    # --- Create Dummy Knowledge File ---
    KNOWLEDGE_FILE = "company_policy.txt"
    knowledge_file_path = os.path.join(supervisor.workspace_dir, KNOWLEDGE_FILE)
    if not os.path.exists(knowledge_file_path):
        print(f"\n--- Creating Dummy Knowledge File: {KNOWLEDGE_FILE} ---")
        policy_content = """
Company Policy Document - v1.2

Remote Work:
Employees in Engineering and Marketing departments are eligible for full remote work.
Sales department requires hybrid work (3 days in office).
HR department requires full-time in-office presence.
All remote employees must maintain core working hours of 10 AM to 4 PM local time.

Vacation Policy:
Standard vacation allowance is 20 days per year, accrued monthly.
Unused vacation days up to 5 can be carried over to the next year.
Requests must be submitted via the HR portal at least 2 weeks in advance.

Project Phoenix:
This is a high-priority project for Q3/Q4.
Lead Engineer: Charlie. Project Manager: David.
Weekly status reports are due every Friday by 5 PM EST.
Key deliverable: Beta release by November 15th.
"""
        try:
            with open(knowledge_file_path, "w") as f:
                f.write(policy_content)
            print(f"Knowledge file created at {knowledge_file_path}")
        except Exception as e:
            print(f"Error creating knowledge file: {e}")

    # --- Add Knowledge using the Tool (Run once interactively or ensure it's done) ---
    if os.path.exists(knowledge_file_path):
        # Check if knowledge might already be added (simple check based on Mem0)
        # In a real system, you'd track ingestions more robustly.
        mem_search = supervisor.mem0_memory.search(query=f"Ingested.*{KNOWLEDGE_FILE}", user_id="supervisor_system", limit=1)
        if not mem_search:
            print("\n--- Task: Add Knowledge to KB (First time setup) ---")
            task_add_knowledge = f"Please add the document '{KNOWLEDGE_FILE}' to the knowledge base."
            result_add_knowledge = supervisor.run(task_add_knowledge, user_id="admin_user")
            print("\nAdd Knowledge Task Result:", result_add_knowledge.get('final_output', result_add_knowledge)) # Show final synthesized output
        else:
            print(f"\n--- Knowledge file '{KNOWLEDGE_FILE}' likely already ingested. Skipping addition. ---")
    else:
        print("\nSkipping knowledge addition task (knowledge file not found).")


    # --- CHOOSE EXECUTION MODE ---
    # Option 1: Run Autonomous Scheduler (Blocks interactive use)
    # Option 2: Run Interactive Chat Examples (Comment out scheduler below)

    RUN_AUTONOMOUS_SCHEDULER = True # Set to False to run interactive examples instead

    if RUN_AUTONOMOUS_SCHEDULER:
        print("\n--- Starting Autonomous Scheduler Mode ---")
        print("NOTE: This mode will run scheduled tasks indefinitely.")
        print("      Interactive chat examples below will be skipped.")
        print("      Press Ctrl+C to stop the scheduler.")

        # --- Define Scheduled Tasks ---
        print("\n--- Defining Scheduled Tasks ---")

        def run_daily_summary_task():
            print(f"\n[{datetime.now()}] Triggering Daily Summary Task...")
            # Example: Define the task as a specific prompt
            task_prompt = "Query the database for the total number of employees in each department and list them."
            supervisor._execute_autonomous_task(task_prompt, task_id="daily_dept_summary")

        def run_knowledge_check_task():
             print(f"\n[{datetime.now()}] Triggering Knowledge Check Task...")
             # Example: Ask a question that requires the KB
             task_prompt = "What is the company policy on remote work for the Sales department?"
             supervisor._execute_autonomous_task(task_prompt, task_id="kb_remote_work_check")

        # --- Schedule the Tasks ---
        # schedule.every().day.at("08:00").do(run_daily_summary_task)
        # schedule.every().monday.at("09:00").do(supervisor.perform_weekly_forecast) # If using specific methods
        schedule.every(1).minutes.do(run_daily_summary_task) # Example: Run summary every minute for testing
        schedule.every(90).seconds.do(run_knowledge_check_task) # Example: Run KB check every 90s for testing

        print("\n--- Starting Scheduler ---")
        print(f"Scheduled tasks: {schedule.get_jobs()}")

        # --- Run the Scheduler Loop ---
        try:
            while True:
                schedule.run_pending()
                time.sleep(1) # Check every second
        except KeyboardInterrupt:
            print("\nScheduler stopped by user.")

        # --- Event-Driven Trigger Note ---
        # To implement event-driven triggers (e.g., reacting to new files),
        # you would typically run a separate script (like the watchdog example)
        # that imports and uses this initialized 'supervisor' instance.
        # That script would call `supervisor._execute_autonomous_task(...)`
        # when an event occurs. It's generally not mixed directly into this
        # main scheduling loop unless using a more advanced framework like asyncio or APScheduler.

        # --- Goal-Oriented Note ---
        # For more complex autonomous behavior where the agent needs to plan
        # and adapt (e.g., "Monitor website X and if it's down for 5 mins, try Y, then report"),
        # consider restructuring the agent using LangGraph. This involves defining
        # states, nodes (functions for steps), and edges (transitions) to create
        # a state machine that pursues the goal.

    else:
        print("\n--- Starting Interactive Chat Mode ---")
        print("NOTE: Autonomous scheduler is disabled.")

        # --- Ask Questions (Leveraging RAG) ---
        print("\n--- Task: Ask question requiring KB ---")
        task_rag1 = "What is the vacation policy regarding carry-over days?"
        result_rag1 = supervisor.run(task_rag1, user_id="employee_alice")
        print("\nKB Question 1 Result:", result_rag1.get('final_output', result_rag1))

        print("\n--- Task: Ask another question requiring KB ---")
        task_rag2 = "Who is the project manager for Project Phoenix?"
        result_rag2 = supervisor.run(task_rag2, user_id="employee_bob")
        print("\nKB Question 2 Result:", result_rag2.get('final_output', result_rag2))

        print("\n--- Task: Ask question combining KB and potential DB ---")
        task_rag_db = "What is Charlie's salary and role on Project Phoenix?" # Requires DB for salary, KB for role
        result_rag_db = supervisor.run(task_rag_db, user_id="manager_dave")
        print("\nCombined Question Result:", result_rag_db.get('final_output', result_rag_db))

        # --- Run other tasks (Keep existing examples if desired) ---
        # Example: SQL Query
        if DB_URI:
            print("\n--- Task: SQL Query ---")
            task_sql = "Who are the employees in the Engineering department?"
            result_sql = supervisor.run(task_sql, user_id="hr_manager")
            print("\nSQL Task Result:", result_sql.get('final_output', result_sql))

        # Example: Forecasting (if data exists)
        TIME_SERIES_DATA_FILE = "sample_ts_data.csv" # Ensure this matches earlier creation
        ts_file_path = os.path.join(supervisor.workspace_dir, TIME_SERIES_DATA_FILE)
        if os.path.exists(ts_file_path):
            print("\n--- Task: Time Series Forecasting ---")
            task_forecast = f"Forecast 'CallVolume' from '{TIME_SERIES_DATA_FILE}' for 7 days. Time column is 'Date', frequency is 'D'."
            result_forecast = supervisor.run(task_forecast, user_id="ops_manager")
            print("\nForecasting Task Result:", result_forecast.get('final_output', result_forecast))
        else:
            # Create dummy data if missing for demo
            print(f"\n--- Creating dummy time series data: {TIME_SERIES_DATA_FILE} ---")
            try:
                dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
                # Simple sine wave + trend + noise
                volume = (np.sin(np.arange(90) * 2 * np.pi / 7) * 10 +
                          np.linspace(50, 80, 90) +
                          np.random.normal(0, 5, 90))
                ts_df = pd.DataFrame({'Date': dates, 'CallVolume': volume.astype(int)})
                ts_df.to_csv(ts_file_path, index=False)
                print(f"Dummy file '{TIME_SERIES_DATA_FILE}' created in workspace.")
                print("\n--- Task: Time Series Forecasting (with dummy data) ---")
                task_forecast = f"Forecast 'CallVolume' from '{TIME_SERIES_DATA_FILE}' for 7 days. Time column is 'Date', frequency is 'D'."
                result_forecast = supervisor.run(task_forecast, user_id="ops_manager")
                print("\nForecasting Task Result:", result_forecast.get('final_output', result_forecast))
            except Exception as e:
                print(f"Error creating dummy TS data or running forecast: {e}")


    print("\n--- Agent Execution Finished ---")
    print("Note: This is a demo. In a real-world scenario, you would have more robust error handling, logging, and possibly a UI for interaction.")