
import logging
import os
import uuid
import asyncio
from typing import List, Dict, Any, Optional

# --- Core Libraries ---
import chromadb
from chromadb.utils import embedding_functions
from mem0 import Memory

# --- LangChain & LangGraph ---
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool, BaseTool
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.agents.agent_toolkits import FileManagementToolkit

# --- Project Imports ---
# Assuming this file is in core/, agents are in agents/, utils are in utils/
from ..agents.base_agent import BaseAgent # If BaseAgent is needed by tools
from ..agents.sql_agent import SQLAgent
from ..agents.visualization_agent import VisualizationAgent
from ..agents.modeling_agent import ModelingAgent
from ..agents.forecasting_agent import ForecastingAgent
from ..agents.reporting_agent import ReportingAgent
from ..utils import config
from ..utils.document_processing import load_and_split_documents
# Assuming calculate_forecast_metrics is moved
try:
    from ..utils.time_series_utils import calculate_forecast_metrics
except ImportError:
    logging.warning("Time series metrics calculation function not found in utils. Assuming basic fallback.")


class CoreAgentServices:
    """
    Initializes and holds shared resources and capabilities for agent handlers.
    """
    def __init__(self, verbose: bool = False):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Core Agent Services...")
        self.verbose = verbose

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

        # --- Initialize File Toolkit ---
        self.fm_toolkit = FileManagementToolkit(root_dir=self.workspace_dir)

        # --- Create Tools List (pointing to agent methods/wrappers) ---
        self.tools = self._get_tools()

        # --- Create Shared Agent Executor ---
        self.agent_executor = self._create_agent_executor()

        self.logger.info("Core Agent Services Initialized Successfully.")

    # --- Initialization Helpers ---
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

    # --- Async Wrappers for Sync Operations ---
    async def _run_sync_in_thread(self, func, *args, **kwargs):
        """Runs a synchronous function in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs)) # None uses default ThreadPoolExecutor

    # --- Tool Wrappers (Async) ---
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

    async def _add_knowledge_tool_async(self, source_path: str) -> str:
        self.logger.info(f"--- Knowledge Addition Tool --- Source path: {source_path}")
        try:
            documents = await self._run_sync_in_thread(load_and_split_documents, source_path, self.workspace_dir)
            if not documents: return f"Failed to load/split documents from '{source_path}'."
            await self.add_knowledge(documents) # Use the RAG helper
            return f"Successfully processed and added knowledge from '{source_path}'."
        except Exception as e: self.logger.error(f"Error in knowledge addition tool: {e}", exc_info=self.verbose); return f"Error adding knowledge: {e}"

    # --- Tool List Creation ---
    def _get_tools(self) -> List[BaseTool]:
        all_tools = []
        # File Management Tools (Sync - Executor handles threading)
        try:
            all_tools.extend(self.fm_toolkit.get_tools())
            self.logger.debug(f"Loaded File Tools: {[t.name for t in self.fm_toolkit.get_tools()]}")
        except Exception as e: self.logger.warning(f"File tools initialization failed: {e}", exc_info=self.verbose)

        # Add other tools with async wrappers
        if self.sql_agent: all_tools.append(tool(name="query_sql_database", func=self._run_sql_query_tool_async, description="Query SQL DB with natural language.", coroutine=self._run_sql_query_tool_async))
        if self.viz_agent: all_tools.append(tool(name="create_visualization", func=self._run_visualization_tool_async, description=f"Create plot from data (path/CSV string in '{self.workspace_dir}'), save as image. Args: data_source, request, output_filename.", coroutine=self._run_visualization_tool_async))
        if self.modeling_agent: all_tools.append(tool(name="run_modeling_task", func=self._run_modeling_tool_async, description=f"Train/evaluate ML model (classification/regression) from data (path/CSV string in '{self.workspace_dir}'). Args: data_source, request.", coroutine=self._run_modeling_tool_async))
        if self.forecasting_agent: all_tools.append(tool(name="run_time_series_forecast", func=self._run_forecasting_tool_async, description=f"Generate time series forecast using SARIMAX. Args: data_source (path/CSV string in '{self.workspace_dir}'), request.", coroutine=self._run_forecasting_tool_async))
        if self.reporting_agent: all_tools.append(tool(name="generate_pdf_report", func=self._run_reporting_tool_async, description=f"Generate a PDF report from data (path/CSV string in '{self.workspace_dir}') based on a request. Args: data_source, request.", coroutine=self._run_reporting_tool_async))

        all_tools.append(tool(name="add_knowledge_base_document", func=self._add_knowledge_tool_async, description=f"Add knowledge from a document or directory (in workspace '{self.workspace_dir}') to the knowledge base. Args: source_path.", coroutine=self._add_knowledge_tool_async))

        self.logger.info(f"Assembled Tools: {[t.name for t in all_tools]}")
        return all_tools

    # --- Agent Executor Creation ---
    def _create_agent_executor(self) -> Optional[AgentExecutor]:
        agent_type = AgentType.OPENAI_FUNCTIONS
        self.logger.info(f"Creating Agent Executor ({agent_type})")
        if not self.tools: self.logger.error("Cannot create agent executor with no tools."); return None
        try:
            # Prompt defined within the ChatHandler now, executor is more generic
            agent_executor = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=agent_type,
                verbose=self.verbose,
                handle_parsing_errors=True,
                max_iterations=15,
                early_stopping_method="generate",
            )
            self.logger.debug("Agent executor created successfully.")
            return agent_executor
        except Exception as e: self.logger.error(f"Error initializing agent executor: {e}", exc_info=self.verbose); return None

    # --- RAG/Memory Async Helpers ---
    async def add_knowledge(self, documents: List): # Takes LangchainDocument list
        """Adds document chunks to ChromaDB."""
        if not documents: self.logger.warning("No documents provided to add_knowledge."); return
        ids = [str(uuid.uuid4()) for _ in documents]
        contents = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        try:
            self.logger.info(f"Adding {len(contents)} chunks to ChromaDB...")
            await self._run_sync_in_thread(self.collection.add, documents=contents, metadatas=metadatas, ids=ids)
            self.logger.info(f"Successfully added {len(contents)} chunks.")
        except Exception as e: self.logger.error(f"Error adding documents to ChromaDB: {e}", exc_info=self.verbose)

    async def retrieve_context(self, query: str, n_results: int = config.RAG_RESULTS_COUNT) -> List[str]:
        """Queries ChromaDB for relevant document chunks."""
        try:
            results = await self._run_sync_in_thread(
                self.collection.query, query_texts=[query], n_results=n_results, include=['documents']
            )
            if results and results.get('documents') and results['documents'][0]:
                self.logger.debug(f"Retrieved {len(results['documents'][0])} context chunks.")
                return results['documents'][0]
            else: self.logger.debug("No relevant context found."); return []
        except Exception as e: self.logger.error(f"Error querying ChromaDB: {e}", exc_info=self.verbose); return []

    async def retrieve_memory(self, query: str, user_id: str, limit: int = config.MEM0_RESULTS_COUNT) -> List[Dict]:
        """Retrieves relevant memories from Mem0."""
        try:
            relevant_memories = await self._run_sync_in_thread(
                self.mem0_memory.search, query=query, user_id=user_id, limit=limit
            )
            self.logger.debug(f"Retrieved {len(relevant_memories)} memories for user {user_id}.")
            return relevant_memories or []
        except Exception as e: self.logger.error(f"Could not retrieve memories from Mem0: {e}", exc_info=self.verbose); return []

    async def add_memory(self, text: str, user_id: str, agent_id: str):
        """Adds an interaction to Mem0."""
        try:
            await self._run_sync_in_thread(self.mem0_memory.add, text=text, user_id=user_id, agent_id=agent_id)
            self.logger.debug(f"Saved interaction to Mem0: User={user_id}, Agent={agent_id}.")
        except Exception as e: self.logger.error(f"Could not save interaction to Mem0: {e}", exc_info=self.verbose)
