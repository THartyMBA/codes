# agents/visualization_agent.py

import logging
import os
import io
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np # Often used with pandas/plotting
import matplotlib.pyplot as plt
import seaborn as sns

# --- LangChain Components ---
from langchain_core.prompts import ChatPromptTemplate, SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

# --- Project Imports ---
from .base_agent import BaseAgent # Import the base class

# Logger setup is handled by the BaseAgent's __init__
# We just need to use self.logger

class VisualizationAgent(BaseAgent):
    """
    Handles the creation of visualizations based on data and requests.
    Inherits common initialization (LLM, workspace, logger) and utilities (like _load_data) from BaseAgent.
    """
    def __init__(self, llm: Any, workspace_dir: str, verbose: bool = False):
        """
        Initializes the VisualizationAgent.

        Args:
            llm: The language model instance.
            workspace_dir: The absolute path to the agent's workspace directory.
                           This is required for saving plots and potentially loading data.
            verbose: If True, enable more detailed logging.
        """
        # Call BaseAgent's init first
        super().__init__(llm=llm, workspace_dir=workspace_dir, verbose=verbose)

        # Specific check: VisualizationAgent absolutely needs a workspace
        if not self.workspace_dir:
            self.logger.critical("Initialization failed: VisualizationAgent requires a valid workspace_dir.")
            raise ValueError("VisualizationAgent requires a valid workspace_dir.")

        # Ensure plotting libraries are available (optional check)
        try:
            import matplotlib, seaborn
            self.logger.debug("Matplotlib and Seaborn libraries are available.")
        except ImportError:
             self.logger.critical("Matplotlib or Seaborn not found. Please install them (`pip install matplotlib seaborn`).")
             raise ImportError("VisualizationAgent requires matplotlib and seaborn.")

        self.logger.info("VisualizationAgent specific setup complete.")


    def _parse_visualization_request(self, request: str) -> Optional[Dict[str, Any]]:
        """Uses LLM to parse natural language request into plotting parameters."""
        self.logger.debug(f"Parsing visualization request: '{request[:100]}...'")
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
            self.logger.debug(f"LLM parsed visualization parameters: {parsed_params}")
            # Basic validation
            if not isinstance(parsed_params, dict) or 'plot_type' not in parsed_params:
                 self.logger.warning("LLM parsing failed to return valid JSON with plot_type.")
                 return None
            return parsed_params
        except Exception as e:
            self.logger.error(f"Error parsing visualization request with LLM: {e}", exc_info=self.verbose)
            return None

    # NOTE: _load_data method is now inherited from BaseAgent. No need to redefine it here.

    def generate_plot(self, data_source: str, request: str, output_filename: str) -> str:
        """
        Generates a plot based on the request and saves it to the workspace.

        Args:
            data_source: Path to the data file (e.g., 'data.csv') relative to the workspace,
                         an absolute path, or the raw data as a CSV string.
            request: Natural language description of the plot.
                     (e.g., "Create a scatter plot of salary vs experience, colored by department")
            output_filename: The desired name for the output plot file (e.g., 'plot.png').
                             Should end with a valid image extension (.png, .jpg, .svg).

        Returns:
            A message indicating success (including the output path) or failure.
        """
        self.logger.info(f"--- Generating Visualization ---")
        self.logger.info(f"Data Source: {data_source}")
        self.logger.info(f"Request: {request}")
        self.logger.info(f"Output Filename: {output_filename}")

        # 1. Parse Request
        params = self._parse_visualization_request(request)
        if not params:
            error_msg = f"Error: Could not parse the visualization request: '{request}'"
            self.logger.error(error_msg)
            return error_msg

        # 2. Load Data (using inherited method)
        df = self._load_data(data_source)
        if df is None:
            # Error already logged by _load_data
            return f"Error: Could not load data from source: '{data_source}'"
        if df.empty:
            error_msg = f"Error: Data loaded from '{data_source}' is empty."
            self.logger.error(error_msg)
            return error_msg

        self.logger.debug(f"Data loaded successfully for plotting. Columns: {df.columns.tolist()}")

        # 3. Validate Columns specified in parsed parameters
        required_cols = [params.get('x_col'), params.get('y_col'), params.get('color_col'), params.get('size_col'), params.get('text_col')]
        missing_cols = [col for col in required_cols if col and col not in df.columns]
        if missing_cols:
            error_msg = f"Error: Column(s) requested for plotting not found in the data: {missing_cols}"
            self.logger.error(error_msg)
            return error_msg

        # 4. Generate Plot
        fig = None # Initialize figure variable
        try:
            fig = plt.figure(figsize=(10, 6)) # Standard figure size
            plot_type = params.get('plot_type', '').lower()
            sns.set_theme(style="whitegrid") # Apply a nice theme

            self.logger.debug(f"Attempting to create plot of type: '{plot_type}'")

            # --- Plotting Logic (using seaborn where appropriate) ---
            if plot_type == 'scatter':
                sns.scatterplot(data=df, x=params.get('x_col'), y=params.get('y_col'),
                                hue=params.get('color_col'), size=params.get('size_col'))
            elif plot_type == 'line':
                x_col = params.get('x_col')
                # Attempt to sort by x-axis if it's numeric or datetime for sensible line plot
                if x_col and x_col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                         df = df.sort_values(by=x_col)
                         self.logger.debug(f"Sorted data by datetime column '{x_col}' for line plot.")
                    elif pd.api.types.is_numeric_dtype(df[x_col]):
                         df = df.sort_values(by=x_col)
                         self.logger.debug(f"Sorted data by numeric column '{x_col}' for line plot.")
                sns.lineplot(data=df, x=x_col, y=params.get('y_col'), hue=params.get('color_col'))
            elif plot_type == 'bar':
                x_col = params.get('x_col')
                y_col = params.get('y_col')
                hue_col = params.get('color_col')
                agg_func = params.get('aggregate_func')

                if agg_func == 'count' and x_col:
                     sns.countplot(data=df, x=x_col, hue=hue_col)
                     params['ylabel'] = params.get('ylabel') or 'Count' # Auto-set ylabel if countplot
                elif x_col and y_col:
                     estimator = agg_func if agg_func in ['mean', 'median', 'sum'] else 'mean' # Default to mean
                     self.logger.debug(f"Using estimator '{estimator}' for bar plot.")
                     # Handle potential categorical estimator error explicitly
                     try:
                         sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, estimator=estimator)
                     except TypeError as te:
                          if "could not convert string to float" in str(te) and estimator != 'count':
                               self.logger.error(f"TypeError during barplot aggregation. Is '{y_col}' numeric? Error: {te}")
                               raise ValueError(f"Cannot aggregate non-numeric column '{y_col}' with function '{estimator}'.") from te
                          else: raise # Re-raise other TypeErrors
                else:
                    raise ValueError("Bar plot requires at least 'x_col', and either 'y_col' or aggregate_func='count'.")

            elif plot_type == 'hist' or plot_type == 'histogram':
                x_col_hist = params.get('x_col')
                if not x_col_hist: raise ValueError("Histogram requires 'x_col'.")
                sns.histplot(data=df, x=x_col_hist, hue=params.get('color_col'), kde=True) # Add KDE curve
            elif plot_type == 'box' or plot_type == 'boxplot':
                if not params.get('x_col') and not params.get('y_col'): raise ValueError("Box plot requires 'x_col' or 'y_col'.")
                sns.boxplot(data=df, x=params.get('x_col'), y=params.get('y_col'), hue=params.get('color_col'))
            elif plot_type == 'heatmap':
                 # Heatmaps typically need data in a matrix format or require pivoting
                 # This is a simplified version assuming numeric columns for correlation
                 numeric_df = df.select_dtypes(include=np.number)
                 if numeric_df.shape[1] < 2:
                      raise ValueError("Heatmap (correlation) requires at least two numeric columns.")
                 correlation_matrix = numeric_df.corr()
                 sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f") # Format annotations
                 params['title'] = params.get('title') or 'Correlation Heatmap'
            else:
                raise ValueError(f"Unsupported plot type requested: '{plot_type}'. Supported types: scatter, line, bar, hist, box, heatmap.")

            # 5. Customize Plot
            plot_title = params.get('title') or f"{plot_type.capitalize()} Plot"
            plt.title(plot_title)
            if params.get('xlabel'): plt.xlabel(params.get('xlabel'))
            if params.get('ylabel'): plt.ylabel(params.get('ylabel'))
            plt.tight_layout() # Adjust layout to prevent labels overlapping

            # 6. Save Plot
            # Ensure output filename has a valid extension
            valid_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
            if not any(output_filename.lower().endswith(ext) for ext in valid_extensions):
                original_filename = output_filename
                output_filename += '.png' # Default to PNG
                self.logger.warning(f"Output filename '{original_filename}' lacked valid extension, defaulting to '{output_filename}'")

            # Construct full path within the workspace
            output_path = os.path.join(self.workspace_dir, output_filename)
            plt.savefig(output_path)
            self.logger.info(f"Visualization saved successfully to: {output_path}")
            success_message = f"Successfully created visualization '{plot_title}' and saved it to '{output_path}'"
            return success_message

        except Exception as e:
            error_msg = f"Error generating plot: {e}"
            self.logger.error(error_msg, exc_info=self.verbose)
            return error_msg # Return error message to the supervisor
        finally:
             # Ensure plot is closed to free memory, even if errors occurred
             if fig:
                 plt.close(fig)
                 self.logger.debug("Plot figure closed.")

