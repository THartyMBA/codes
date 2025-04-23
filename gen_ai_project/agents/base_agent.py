# agents/base_agent.py

import logging
import os
import io
from typing import Optional, Any
import pandas as pd
import numpy as np # Often needed alongside pandas
from abc import ABC # Optional: Use Abstract Base Classes if you want to enforce methods

# Assuming logging is set up in main.py or via utils.logging_setup
# Each agent instance will get its own named logger
# logger = logging.getLogger(__name__) # Don't configure here, just get logger in subclasses

class BaseAgent(ABC): # Inherit from ABC if using @abstractmethod
    """
    Abstract base class for specialized agents.

    Provides common initialization for LLM, workspace, logging,
    and potentially shared utility methods.
    """

    def __init__(
        self,
        llm: Any, # Expecting an LLM instance (e.g., ChatOllama)
        workspace_dir: Optional[str] = None, # Workspace is optional for some agents (like SQL)
        verbose: bool = False,
        **kwargs # Allow for extra arguments if needed by specific base initializations later
    ):
        """
        Initializes the base agent.

        Args:
            llm: The language model instance used by the agent.
            workspace_dir: The absolute path to the agent's workspace directory.
                           Required by agents dealing with files.
            verbose: If True, enable more detailed logging output.
            **kwargs: Additional keyword arguments.
        """
        self.llm = llm
        self.workspace_dir = workspace_dir
        self.verbose = verbose
        # Each subclass instance gets its own logger named after its class
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.workspace_dir:
            # Ensure workspace exists if provided (though config.py might also do this)
            try:
                os.makedirs(self.workspace_dir, exist_ok=True)
                self.logger.debug(f"Workspace directory: {self.workspace_dir}")
            except OSError as e:
                 self.logger.error(f"Failed to create workspace directory '{self.workspace_dir}': {e}")
                 # Depending on agent needs, might want to raise error here if workspace is critical
        elif self.__class__.__name__ not in ["SQLAgent"]: # Example: Warn if workspace missing for file-based agents
             self.logger.warning(f"Workspace directory not provided for {self.__class__.__name__}. File operations may fail.")


        self.logger.info(f"{self.__class__.__name__} initialized.")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG) # Allow DEBUG messages if verbose
            self.logger.debug("Verbose logging enabled for this agent instance.")

    # --- Optional: Define abstract methods if ALL agents MUST implement them ---
    # from abc import abstractmethod
    # @abstractmethod
    # def run(self, *args, **kwargs) -> Any:
    #     """Main execution method for the agent (adapt signature as needed)."""
    #     pass

    # --- Shared Utility Methods ---

    def _load_data(self, data_source: str) -> Optional[pd.DataFrame]:
        """
        Loads data into a pandas DataFrame from various sources.
        Handles file paths (relative to workspace or absolute) and CSV strings.
        Supports CSV and Excel files.

        Args:
            data_source: Path to the data file (e.g., 'data.csv') relative to the workspace,
                         an absolute path, or the raw data as a CSV string.

        Returns:
            A pandas DataFrame, or None if loading fails.
        """
        if not data_source:
            self.logger.error("Data source cannot be empty.")
            return None

        path = data_source
        is_string_data = False

        # Resolve relative paths against the workspace directory
        if self.workspace_dir and not os.path.isabs(path):
            potential_path = os.path.join(self.workspace_dir, path)
            if os.path.exists(potential_path):
                path = potential_path
            elif ',' in data_source and '\n' in data_source: # Heuristic for CSV string
                 is_string_data = True
                 self.logger.debug("Detected data source as potential string data.")
            # else: path remains the original relative path, might fail later if not found

        elif not os.path.exists(path) and ',' in data_source and '\n' in data_source:
             is_string_data = True
             self.logger.debug("Detected data source as potential string data (absolute path not found).")


        self.logger.debug(f"Attempting to load data from source: '{path if not is_string_data else 'String Data'}'.")

        try:
            if is_string_data:
                self.logger.info("Loading data from string (assuming CSV format).")
                data_io = io.StringIO(data_source)
                df = pd.read_csv(data_io)
            elif os.path.exists(path):
                if path.lower().endswith('.csv'):
                    self.logger.info(f"Loading data from CSV file: {path}")
                    df = pd.read_csv(path)
                elif path.lower().endswith(('.xls', '.xlsx')):
                    self.logger.info(f"Loading data from Excel file: {path}")
                    df = pd.read_excel(path)
                else:
                    self.logger.error(f"Unsupported file type: {path}. Only .csv and .xlsx/.xls are supported by default _load_data.")
                    return None
            else:
                self.logger.error(f"Data source file not found: {path}")
                return None

            self.logger.info(f"Data loaded successfully. Shape: {df.shape}. Columns: {df.columns.tolist()}")
            if df.empty:
                self.logger.warning("Loaded DataFrame is empty.")
            return df

        except Exception as e:
            self.logger.error(f"Error loading data from source '{data_source}': {e}", exc_info=self.verbose) # Show traceback if verbose
            return None

    # Add other common utilities here if identified, e.g.,
    # def _save_output(self, data: Any, filename: str): ...
    # def _log_error(self, message: str, error: Exception): ...

