# core/core_services.py

import logging
import os
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Callable # Added Callable

# --- Core Libraries ---
import chromadb
from chromadb.utils import embedding_functions
from mem0 import Memory

# --- LangChain & LangGraph ---
from langchain_community.chat_models import ChatOllama
# Removed unused prompt/parser imports here, they belong in handlers
from langchain_core.tools import tool, BaseTool
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.agents.agent_toolkits import FileManagementToolkit

# --- Project Imports ---
# Assuming this file is in core/, agents are in agents/, utils are in utils/, handlers are in handlers/
from ..agents.base_agent import BaseAgent # If BaseAgent is needed by tools
from ..agents.sql_agent import SQLAgent
from ..agents.visualization_agent import VisualizationAgent
from ..agents.modeling_agent import ModelingAgent
from ..agents.forecasting_agent import ForecastingAgent
from ..agents.reporting_agent import ReportingAgent
from ..handlers.knowledge_handler import KnowledgeHandler # Added
from ..handlers.goal_handler import GoalHandler # Added
from ..handlers.pipeline_handler import PipelineHandler # Added
from ..handlers.monitoring_handler import MonitoringHandler # Added
from ..handlers.admin_handler import AdminHandler # Added
# Event and Scheduler handlers are typically managed at the API/run level, but Admin might need refs
from ..handlers.event_handler import EventHandler
from ..handlers.scheduler_handler import SchedulerHandler
from ..utils import config
from ..utils.document_processing import load_and_split_documents
# Assuming calculate_forecast_metrics is moved
try:
    from ..utils.time_series_utils import calculate_forecast_metrics
except ImportError:
    logging.warning("Time series metrics calculation function not found in utils. Assuming basic fallback.")


class CoreAgentServices:
    """
    Initializes and holds shared resources, tool agents, the core agent executor,
    and instances of operational handlers.
    """
    def __init__(self, event_queue: asyncio.Queue, verbose: bool = False):
        """
        Initializes CoreAgentServices.

        Args:
            event_queue: The shared asyncio.Queue for event handling.
            verbose: Enable verbose logging.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Core Agent Services...")
        self.verbose = verbose
        self.event_queue = event_queue # Store event queue reference

        # --- Initialize Shared Resources ---
        self.llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=config.DEFAULT_LLM_TEMPERATURE)
        self.workspace_dir = config.WORKSPACE_DIR # Absolute path from config
        self.embedding_func = self._initialize_embedding_function()
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=self.embedding_func
        )
        self.mem0_memory = Memory()
        self.logger.info(f"LLM, ChromaDB ({config.COLLECTION_NAME}), Mem0 initialized.")

        # --- Initialize Tool Agents (passing shared resources) ---
        self.sql_agent = self._initialize_sql_agent()
        self.viz_agent = self._initialize_visualization_agent()
        self.modeling_agent = self._initialize_modeling_agent()
        self.forecasting_agent = self._initialize_forecasting_agent()
        self.reporting_agent = self._initialize_reporting_agent()
        self.logger.info("Tool agents initialized.")

        # --- Initialize File Toolkit ---
        self.fm_toolkit = FileManagementToolkit(root_dir=self.workspace_dir)
        self.logger.info("File Management Toolkit initialized.")

        # --- Initialize Handlers (passing self for services access) ---
        # Order might matter if handlers depend on each other during init, but usually not.
        self.knowledge_handler = KnowledgeHandler(self)
        self.goal_handler = GoalHandler(self)
        self.pipeline_handler = PipelineHandler(self)
        # Monitoring handler needs the event queue
        self.monitoring_handler = MonitoringHandler(self, self.event_queue, check_interval_seconds=90)
        # Admin handler might need references to other handlers for status reporting
        # Note: Event and Scheduler handlers are often started/managed externally (e.g., api.py)
        # but we pass placeholder references if AdminHandler needs them.
        # These external handlers would also need 'self' (CoreAgentServices) passed to them.
        self.admin_handler = AdminHandler(
            services=self,
            # Pass None for handlers managed externally if Admin doesn't strictly need them at init
            scheduler_handler=None, # Placeholder
            event_handler=None, # Placeholder
            goal_handler=self.goal_handler,
            pipeline_handler=self.pipeline_handler
        )
        self.logger.info("Operational handlers initialized.")

        # --- Create Tools List (pointing to agent methods/wrappers) ---
        self.tools = self._get_tools()

        # --- Create Shared Agent Executor ---
        # This executor is used by ChatHandler, EventHandler, SchedulerHandler etc.
        self.agent_executor = self._create_agent_executor()

        self.logger.info("Core Agent Services Initialized Successfully.")

    # --- Initialization Helpers (Mostly unchanged) ---
    def _initialize_embedding_function(self):
        self.logger.info(f"Initializing embedding function: {config.EMBEDDING_MODEL_NAME}")
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL_NAME)

    def _initialize_sql_agent(self) -> Optional[SQLAgent]:
        if not config.DB_URI: return None
        try: return SQLAgent(llm=self.llm, db_uri=config.DB_URI, verbose=self.verbose)
        except Exception as e: self.logger.error(f"SQL Agent initialization failed: {e}", exc_info=self.verbose); return None

    def _initialize_visualization_agent(self) -> Optional[VisualizationAgent]:
        try: return VisualizationAgent(llm=self.llm, workspace_dir=self.workspace_dir, verbose=self.verbose)
        except Exception as e: self.logger.error(f"Visualization Agent initialization failed: {e}", exc_info=self.verbose); return None

    def _initialize_modeling_agent(self) -> Optional[ModelingAgent]:
         try: return ModelingAgent(llm=self.llm, workspace_dir=self.workspace_dir, verbose=self.verbose)
         except Exception as e: self.logger.error(f"Modeling Agent initialization failed: {e}", exc_info=self.verbose); return None

    def _initialize_forecasting_agent(self) -> Optional[ForecastingAgent]:
         try: return ForecastingAgent(llm=self.llm, workspace_dir=self.workspace_dir, verbose=self.verbose)
         except Exception as e: self.logger.error(f"Forecasting Agent initialization failed: {e}", exc_info=self.verbose); return None

    def _initialize_reporting_agent(self) -> Optional[ReportingAgent]:
         try: return ReportingAgent(llm=self.llm, workspace_dir=self.workspace_dir, verbose=self.verbose)
         except Exception as e: self.logger.error(f"Reporting Agent initialization failed: {e}", exc_info=self.verbose); return None

    # --- Async Wrappers for Sync Operations (Unchanged) ---
    async def _run_sync_in_thread(self, func, *args, **kwargs):
        """Runs a synchronous function in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    # --- Tool Wrappers (Async - Unchanged, except knowledge add removed) ---
    async def _run_sql_query_tool_async(self, natural_language_query: str) -> str:
        if not self.sql_agent: return "SQL Agent not initialized."
        try:
            response = await self._run_sync_in_thread(self.sql_agent.run, natural_language_query)
            if response.get("error"): return f"SQL Error: {response['error']}\nSQL: {response.get('generated_sql', 'N/A')}"
            return f"Result:\n{response.get('result', 'N/A')}\n\nSQL:\n```sql\n{response.get('generated_sql', 'N/A')}\n```"
        except Exception as e: self.logger.error(f"SQL tool error: {e}", exc_info=self.verbose); return f"SQL tool error: {e}"

    async def _run_visualization_tool_async(self, data_source: str, request: str, output_filename: str) -> str:
        if not self.viz_agent: return "Visualization Agent not initialized."
        try:
            return await self._run_sync_in_thread(self.viz_agent.generate_plot, data_source, request, output_filename)
        except Exception as e: self.logger.error(f"Visualization tool error: {e}", exc_info=self.verbose); return f"Visualization tool error: {e}"

    async def _run_modeling_tool_async(self, data_source: str, request: str) -> str:
        if not self.modeling_agent: return "Modeling Agent not initialized."
        try:
            return await self._run_sync_in_thread(self.modeling_agent.run_modeling_task, data_source, request)
        except Exception as e: self.logger.error(f"Modeling tool error: {e}", exc_info=self.verbose); return f"Modeling tool error: {e}"

    async def _run_forecasting_tool_async(self, data_source: str, request: str) -> str:
        if not self.forecasting_agent: return "Forecasting Agent not initialized."
        try:
            return await self._run_sync_in_thread(self.forecasting_agent.run_forecasting_task, data_source, request)
        except Exception as e: self.logger.error(f"Forecasting tool error: {e}", exc_info=self.verbose); return f"Forecasting tool error: {e}"

    async def _run_reporting_tool_async(self, data_source: str, request: str) -> str:
        if not self.reporting_agent: return "Reporting Agent not initialized."
        try:
            # Reporting agent's run method is already async
            return await self.reporting_agent.run_report_generation(data_source, request)
        except Exception as e: self.logger.error(f"Reporting tool error: {e}", exc_info=self.verbose); return f"Reporting tool error: {e}"

    # --- Wrapper for Knowledge Add Tool ---
    async def _knowledge_add_tool_wrapper(self, source_path: str) -> str:
        """Wrapper for knowledge handler add_source to return string for agent tools."""
        if not self.knowledge_handler: return "Error: Knowledge Handler not available."
        try:
            result = await self.knowledge_handler.add_source(source_path)
            return result.get("message", "An unknown error occurred during knowledge addition.")
        except Exception as e:
            self.logger.error(f"Error in knowledge add tool wrapper: {e}", exc_info=self.verbose)
            return f"Error adding knowledge: {e}"

    # --- Tool List Creation (Updated) ---
    def _get_tools(self) -> List[BaseTool]:
        all_tools = []
        # File Management Tools
        try:
            all_tools.extend(self.fm_toolkit.get_tools())
            self.logger.debug(f"Loaded File Tools: {[t.name for t in self.fm_toolkit.get_tools()]}")
        except Exception as e: self.logger.warning(f"File tools initialization failed: {e}", exc_info=self.verbose)

        # Add other tools with async wrappers
        if self.sql_agent: all_tools.append(tool(name="query_sql_database", func=self._run_sql_query_tool_async, description="Query SQL DB with natural language.", coroutine=self._run_sql_query_tool_async))
        if self.viz_agent: all_tools.append(tool(name="create_visualization", func=self._run_visualization_tool_async, description=f"Create plot from data (path/CSV string in '{self.workspace_dir}'), save as image. Args: data_source, request, output_filename.", coroutine=self._run_visualization_tool_async))
        if self.modeling_agent: all_tools.append(tool(name="run_modeling_task", func=self._run_modeling_tool_async, description=f"Train/evaluate ML model from data (path/CSV string in '{self.workspace_dir}'). Args: data_source, request.", coroutine=self._run_modeling_tool_async))
        if self.forecasting_agent: all_tools.append(tool(name="run_time_series_forecast", func=self._run_forecasting_tool_async, description=f"Generate time series forecast using SARIMAX. Args: data_source (path/CSV string in '{self.workspace_dir}'), request.", coroutine=self._run_forecasting_tool_async))
        if self.reporting_agent: all_tools.append(tool(name="generate_pdf_report", func=self._run_reporting_tool_async, description=f"Generate a PDF report from data (path/CSV string in '{self.workspace_dir}') based on a request. Args: data_source, request.", coroutine=self._run_reporting_tool_async))

        # Updated Knowledge Add Tool
        if self.knowledge_handler:
             # Need a sync wrapper for func if executor doesn't handle coroutine-only tools well
             # Using asyncio.run is generally discouraged within an existing loop,
             # but might be necessary if the agent framework requires a sync func.
             # A better approach is if the framework directly supports async func/coroutine.
             # Assuming the framework handles `coroutine` correctly:
             all_tools.append(tool(
                 name="add_knowledge_base_document",
                 func=None, # Let coroutine handle it
                 description=f"Add knowledge from a document or directory (in workspace '{self.workspace_dir}') to the knowledge base. Args: source_path.",
                 coroutine=self._knowledge_add_tool_wrapper
             ))
             # If sync func is strictly required:
             # sync_wrapper = lambda sp: asyncio.run(self._knowledge_add_tool_wrapper(sp))
             # all_tools.append(tool(..., func=sync_wrapper, coroutine=self._knowledge_add_tool_wrapper))


        self.logger.info(f"Assembled Tools: {[t.name for t in all_tools]}")
        return all_tools

    # --- Agent Executor Creation (Unchanged conceptually) ---
    def _create_agent_executor(self) -> Optional[AgentExecutor]:
        agent_type = AgentType.OPENAI_FUNCTIONS # Or another appropriate type
        self.logger.info(f"Creating Agent Executor ({agent_type})")
        if not self.tools: self.logger.error("Cannot create agent executor with no tools."); return None
        try:
            # The prompt is now primarily the responsibility of the handler using the executor (e.g., ChatHandler)
            # This executor is more generic.
            agent_executor = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=agent_type,
                verbose=self.verbose,
                handle_parsing_errors=True, # Use default error message or a custom function/string
                max_iterations=15,
                early_stopping_method="generate",
            )
            self.logger.debug("Agent executor created successfully.")
            return agent_executor
        except Exception as e: self.logger.error(f"Error initializing agent executor: {e}", exc_info=self.verbose); return None

    # --- RAG/Memory Async Helpers (Removed RAG, kept Memory) ---

    # REMOVED: add_knowledge (moved to KnowledgeHandler)
    # REMOVED: retrieve_context (moved to KnowledgeHandler)

    async def retrieve_memory(self, query: str, user_id: str, limit: int = config.MEM0_RESULTS_COUNT) -> List[Dict]:
        """Retrieves relevant memories from Mem0."""
        # (Implementation remains the same, using _run_sync_in_thread)
        try:
            relevant_memories = await self._run_sync_in_thread(
                self.mem0_memory.search, query=query, user_id=user_id, limit=limit
            )
            self.logger.debug(f"Retrieved {len(relevant_memories)} memories for user {user_id}.")
            return relevant_memories or []
        except Exception as e: self.logger.error(f"Could not retrieve memories from Mem0: {e}", exc_info=self.verbose); return []

    async def add_memory(self, text: str, user_id: str, agent_id: str):
        """Adds an interaction to Mem0."""
        # (Implementation remains the same, using _run_sync_in_thread)
        try:
            await self._run_sync_in_thread(self.mem0_memory.add, text=text, user_id=user_id, agent_id=agent_id)
            self.logger.debug(f"Saved interaction to Mem0: User={user_id}, Agent={agent_id}.")
        except Exception as e: self.logger.error(f"Could not save interaction to Mem0: {e}", exc_info=self.verbose)


