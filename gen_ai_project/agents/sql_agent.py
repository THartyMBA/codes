# agents/sql_agent.py

import logging
from typing import Dict, Any, Optional

# --- Database Interaction ---
from sqlalchemy import create_engine, exc as sqlalchemy_exc
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent, AgentExecutor
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# --- Project Imports ---
from .base_agent import BaseAgent # Import the base class

# Logger setup is handled by the BaseAgent's __init__
# We just need to use self.logger


class SQLAgent(BaseAgent):
    """
    An agent specialized in interacting with a SQL database using LangChain.
    Inherits common initialization and utilities from BaseAgent.
    """
    def __init__(self, llm: Any, db_uri: str, verbose: bool = False):
        """
        Initializes the SQLAgent.

        Args:
            llm: The language model instance.
            db_uri: The SQLAlchemy database connection URI.
            verbose: If True, enable more detailed logging and agent output.
        """
        # Call BaseAgent's init first (workspace_dir is not needed for SQLAgent)
        super().__init__(llm=llm, verbose=verbose)
        self.db_uri = db_uri

        # Specific SQLAgent initialization
        self.db: Optional[SQLDatabase] = None
        self.agent_executor: Optional[AgentExecutor] = None

        try:
            self.db = self._connect_db()
            if self.db:
                self.agent_executor = self._create_sql_agent()
            else:
                 # Error already logged in _connect_db
                 raise ConnectionError("Database connection failed during initialization.")
            self.logger.info("SQLAgent specific setup complete.")
        except Exception as e:
             # Log the final initialization failure
             self.logger.critical(f"SQLAgent initialization failed: {e}", exc_info=self.verbose)
             # Optionally re-raise or handle depending on desired application behavior
             # raise # Re-raise if initialization failure should stop the application


    def _connect_db(self) -> Optional[SQLDatabase]:
        """Connects to the database using the provided URI."""
        if not self.db_uri:
            self.logger.error("Database URI cannot be empty.")
            return None
        try:
            self.logger.info(f"Attempting to connect to database: {self.db_uri.split('@')[-1]}") # Avoid logging credentials
            engine = create_engine(self.db_uri)

            # Test connection - this will raise an error if connection fails
            with engine.connect() as connection:
                 self.logger.debug("Database connection test successful.")

            # Initialize SQLDatabase utility
            # include_tables can be used to limit the scope if needed
            # db = SQLDatabase(engine, include_tables=['employees', 'departments'])
            db = SQLDatabase(engine)

            self.logger.info(f"SQLDatabase utility initialized for dialect: {db.dialect}")
            usable_tables = db.get_usable_table_names()
            self.logger.info(f"Usable Tables detected: {usable_tables}")
            if not usable_tables:
                 self.logger.warning("No usable tables found by SQLDatabase. Agent might struggle to find relevant tables.")
            # self.logger.debug(f"Sample table info (employees): {db.get_table_info(['employees'])}") # Example inspection
            return db
        except sqlalchemy_exc.SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy error connecting to database at {self.db_uri.split('@')[-1]}: {e}", exc_info=self.verbose)
            return None
        except ImportError as e:
             self.logger.error(f"Missing database driver for {self.db_uri}. Error: {e}", exc_info=self.verbose)
             return None
        except Exception as e:
            # Catch any other unexpected errors during connection
            self.logger.error(f"Unexpected error connecting to database at {self.db_uri.split('@')[-1]}: {e}", exc_info=self.verbose)
            return None

    def _create_sql_agent(self) -> Optional[AgentExecutor]:
        """Creates the LangChain SQL agent executor."""
        if not self.db:
            self.logger.error("Cannot create SQL agent without a valid database connection.")
            return None
        try:
            self.logger.debug("Creating SQLDatabaseToolkit...")
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)

            # Context provided to the agent can be crucial
            # (Consider making this configurable or refining it)
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
            """.format(dialect=self.db.dialect, top_k=10) # Example context

            # Choose agent type (OPENAI_FUNCTIONS often works well if LLM supports it)
            agent_type = "openai-tools" # or AgentType.OPENAI_FUNCTIONS

            self.logger.info(f"Creating SQL agent executor with type: {agent_type}")
            agent_executor = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                agent_type=agent_type,
                verbose=self.verbose, # Pass verbose flag to the agent
                handle_parsing_errors=True, # Let agent try to recover from parsing errors
                # prefix=agent_context, # Prefix might be needed depending on agent type
                agent_executor_kwargs={"return_intermediate_steps": True}, # Get intermediate steps for SQL extraction
                max_iterations=10, # Limit iterations to prevent runaway loops
                early_stopping_method="generate" # Stop if agent generates final answer
            )
            self.logger.debug("SQL agent executor created successfully.")
            return agent_executor
        except Exception as e:
            self.logger.error(f"Error creating SQL agent executor: {e}. Ensure the LLM and toolkit are compatible.", exc_info=self.verbose)
            return None

    def run(self, query: str) -> Dict[str, Any]:
        """
        Runs a natural language query against the SQL database using the agent.

        Args:
            query: The natural language query.

        Returns:
            A dictionary containing:
                - "result": The agent's final answer (string).
                - "generated_sql": The SQL query generated (string, best effort extraction).
                - "error": An error message string if an error occurred, otherwise None.
        """
        self.logger.info(f"--- Running SQL Agent Query ---")
        self.logger.info(f"Natural Language Query: {query}")

        if not self.agent_executor:
            self.logger.error("SQL Agent Executor is not available.")
            return {"result": None, "generated_sql": None, "error": "SQL Agent Executor not initialized."}

        generated_sql = "Could not extract SQL."
        response = None # Initialize response to None

        try:
            # Ensure input matches what the specific agent expects (often a dict)
            response = self.agent_executor.invoke({"input": query})

            # --- Extract Generated SQL (Best Effort) ---
            if isinstance(response, dict) and 'intermediate_steps' in response and response['intermediate_steps']:
                for step in reversed(response['intermediate_steps']):
                     # Intermediate steps format can vary, adjust based on observation
                     # Common format: tuple(AgentAction, observation_string)
                     if isinstance(step, tuple) and len(step) > 0:
                         action = step[0]
                         tool_name = getattr(action, 'tool', '').lower()
                         tool_input = getattr(action, 'tool_input', None)

                         # Look for common SQL tool names and inputs
                         if 'sql' in tool_name and tool_input:
                             if isinstance(tool_input, str):
                                 generated_sql = tool_input
                                 break
                             elif isinstance(tool_input, dict) and 'query' in tool_input:
                                 generated_sql = tool_input['query']
                                 break
            else:
                 self.logger.warning("Could not find intermediate steps in agent response to extract SQL.")


            # Clean up SQL potentially wrapped in markdown
            if generated_sql.startswith("```sql"):
                generated_sql = generated_sql.split("```sql\n", 1)[1].split("\n```", 1)[0]

            result = response.get('output', 'Agent did not produce an output.') if isinstance(response, dict) else "Invalid response format."

            self.logger.info(f"Generated SQL (best effort): {generated_sql}")
            self.logger.info(f"Query Result: {result}")
            self.logger.info(f"--- SQL Agent Finished ---")

            return {
                "result": result,
                "generated_sql": generated_sql,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Error during SQL agent execution for query '{query}': {e}", exc_info=self.verbose)
            # Try to get partial info if available from response before error
            output = None
            if isinstance(response, dict):
                output = response.get('output')
            return {
                "result": output, # May be None if error happened early
                "generated_sql": generated_sql, # May still hold value if extracted before error
                "error": str(e)
            }

