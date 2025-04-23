# agents/supervisor_agent.py
#This code file is deprecated in favor of the new 'handlers' module approach, but the code is commented out for reference in case anyone wants to attempt this approach

# import logging
# import os
# import uuid
# import asyncio
# import json
# from datetime import datetime
# from typing import List, Dict, Any, Optional, Sequence

# # --- Core Libraries ---
# import chromadb
# from chromadb.utils import embedding_functions
# from mem0 import Memory

# # --- LangChain & LangGraph ---
# from langchain_community.chat_models import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessage, HumanMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.tools import tool, BaseTool
# from langchain.agents import AgentType, initialize_agent, AgentExecutor
# from langchain.agents.agent_toolkits import FileManagementToolkit
# from langchain_core.runnables import RunnableConfig # For passing config like user_id
# # LangGraph components (placeholders for now, actual graphs would be defined elsewhere)
# # from langgraph.graph import StateGraph, END
# # from langgraph.checkpoint.sqlite import SqliteSaver # Example for persistence

# # --- Scheduling ---
# from apscheduler.schedulers.asyncio import AsyncIOScheduler

# # --- Project Imports ---
# from .base_agent import BaseAgent
# from .sql_agent import SQLAgent
# from .visualization_agent import VisualizationAgent
# from .modeling_agent import ModelingAgent
# from .forecasting_agent import ForecastingAgent
# # Assuming graph definitions might live here or in a dedicated 'graphs' module
# # from ..graphs.monitor_graph import build_monitor_graph, WebsiteMonitorState # Example
# from ..utils import config
# from ..utils.document_processing import load_and_split_documents
# # Assuming calculate_forecast_metrics is moved
# try:
#     from ..utils.time_series_utils import calculate_forecast_metrics
# except ImportError:
#     logging.warning("Time series metrics calculation function not found in utils. Assuming basic fallback.")


# logger = logging.getLogger(__name__) # Get logger named after the module

# class SupervisorAgent(BaseAgent):
#     """
#     An advanced supervisor agent capable of:
#     - Interactive chat with RAG and Memory.
#     - Executing scheduled autonomous tasks.
#     - Reacting to external events via an async queue.
#     - Managing goal-oriented tasks using LangGraph (conceptual).
#     """
#     def __init__(self, model_name: str = config.OLLAMA_MODEL, temperature: float = config.DEFAULT_LLM_TEMPERATURE, db_uri: Optional[str] = config.DB_URI, verbose: bool = False):
#         """
#         Initializes the SupervisorAgent.

#         Args:
#             model_name: Name of the Ollama model to use.
#             temperature: Temperature for LLM generation.
#             db_uri: Database URI for the SQL Agent.
#             verbose: Enable verbose logging for BaseAgent and potentially sub-agents.
#         """
#         # Initialize BaseAgent (handles LLM, workspace, base logger)
#         # Use workspace path from config
#         super().__init__(llm=None, workspace_dir=config.WORKSPACE_DIR, verbose=verbose) # LLM set below
#         self.llm = ChatOllama(model=model_name, temperature=temperature)
#         self.db_uri = db_uri
#         self.logger.info(f"Supervisor LLM initialized: {model_name} (Temp: {temperature})")

#         # --- Core Components ---
#         self.sql_agent_instance = self._initialize_sql_agent()
#         self.visualization_agent_instance = self._initialize_visualization_agent()
#         self.modeling_agent_instance = self._initialize_modeling_agent()
#         self.forecasting_agent_instance = self._initialize_forecasting_agent()

#         # --- RAG Components ---
#         self.embedding_func = self._initialize_embedding_function()
#         self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
#         self.collection = self.chroma_client.get_or_create_collection(
#             name=config.COLLECTION_NAME,
#             embedding_function=self.embedding_func
#         )
#         self.logger.info(f"ChromaDB collection '{config.COLLECTION_NAME}' loaded/created at '{config.CHROMA_DB_PATH}'.")

#         # --- Memory Component ---
#         self.mem0_memory = Memory()
#         self.logger.info("Mem0 initialized.")

#         # --- Tools & Agent Executor ---
#         self.tools = self._get_tools()
#         # Agent Executor needs to be created carefully if tools need async execution
#         self.agent_executor = self._create_agent_executor()

#         # --- Asynchronous Components ---
#         self.scheduler = AsyncIOScheduler()
#         self.event_queue = asyncio.Queue() # Queue for decoupling event handling
#         self.running_goals = {} # Dictionary to track active LangGraph goal instances {goal_id: graph_task}

#         self.logger.info("SupervisorAgent initialization complete.")

#     # --- Initialization Helpers ---
#     def _initialize_embedding_function(self):
#         logger.info(f"Initializing embedding function: {config.EMBEDDING_MODEL_NAME}")
#         # Add logic for OpenAI if needed, based on config.EMBEDDING_MODEL_NAME
#         return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL_NAME)

#     def _initialize_sql_agent(self) -> Optional[SQLAgent]:
#         if not self.db_uri: return None
#         try: return SQLAgent(llm=self.llm, db_uri=self.db_uri, verbose=self.verbose)
#         except Exception as e: logger.error(f"SQL Agent initialization failed: {e}", exc_info=self.verbose); return None

#     def _initialize_visualization_agent(self) -> Optional[VisualizationAgent]:
#         try: return VisualizationAgent(llm=self.llm, workspace_dir=self.workspace_dir, verbose=self.verbose)
#         except Exception as e: logger.error(f"Visualization Agent initialization failed: {e}", exc_info=self.verbose); return None

#     def _initialize_modeling_agent(self) -> Optional[ModelingAgent]:
#          try: return ModelingAgent(llm=self.llm, workspace_dir=self.workspace_dir, verbose=self.verbose)
#          except Exception as e: logger.error(f"Modeling Agent initialization failed: {e}", exc_info=self.verbose); return None

#     def _initialize_forecasting_agent(self) -> Optional[ForecastingAgent]:
#          try: return ForecastingAgent(llm=self.llm, workspace_dir=self.workspace_dir, verbose=self.verbose)
#          except Exception as e: logger.error(f"Forecasting Agent initialization failed: {e}", exc_info=self.verbose); return None

#     # --- RAG/Memory Async Helpers (using asyncio.to_thread for sync libs) ---
#     async def _add_documents_to_chroma_async(self, documents: List[LangchainDocument]):
#         if not documents: logger.warning("No documents provided to add."); return
#         ids = [str(uuid.uuid4()) for _ in documents]
#         contents = [doc.page_content for doc in documents]
#         metadatas = [doc.metadata for doc in documents]
#         try:
#             logger.info(f"Adding {len(contents)} document chunks to ChromaDB '{config.COLLECTION_NAME}'...")
#             # ChromaDB client operations might be synchronous
#             await asyncio.to_thread(self.collection.add, documents=contents, metadatas=metadatas, ids=ids)
#             logger.info(f"Successfully added {len(contents)} chunks.")
#             # Add to Mem0 (assuming Mem0 client is sync)
#             sources_str = ", ".join(list(set(m.get('source', 'unknown') for m in metadatas)))
#             await asyncio.to_thread(
#                 self.mem0_memory.add,
#                 text=f"Ingested {len(contents)} chunks from sources: {sources_str}",
#                 user_id="supervisor_system", agent_id="doc_ingestor"
#             )
#         except Exception as e: logger.error(f"Error adding documents to ChromaDB/Mem0: {e}", exc_info=self.verbose)

#     async def _retrieve_relevant_context_async(self, query: str, n_results: int = config.RAG_RESULTS_COUNT) -> List[str]:
#         try:
#             # ChromaDB query might be synchronous
#             results = await asyncio.to_thread(
#                 self.collection.query, query_texts=[query], n_results=n_results, include=['documents']
#             )
#             if results and results.get('documents') and results['documents'][0]:
#                 logger.debug(f"Retrieved {len(results['documents'][0])} relevant context chunks from ChromaDB.")
#                 return results['documents'][0]
#             else: logger.debug("No relevant documents found in ChromaDB."); return []
#         except Exception as e: logger.error(f"Error querying ChromaDB: {e}", exc_info=self.verbose); return []

#     async def _retrieve_relevant_memories_async(self, query: str, user_id: str, limit: int = config.MEM0_RESULTS_COUNT) -> List[Dict]:
#         try:
#             # Mem0 search might be synchronous
#             relevant_memories = await asyncio.to_thread(
#                 self.mem0_memory.search, query=query, user_id=user_id, limit=limit
#             )
#             if relevant_memories:
#                 logger.debug(f"Retrieved {len(relevant_memories)} relevant memories from Mem0 for user {user_id}.")
#             else:
#                  logger.debug(f"No relevant memories found in Mem0 for user {user_id}.")
#             return relevant_memories or []
#         except Exception as e:
#             logger.error(f"Could not retrieve memories from Mem0: {e}", exc_info=self.verbose)
#             return []

#     async def _add_memory_async(self, text: str, user_id: str, agent_id: str):
#         try:
#             # Mem0 add might be synchronous
#             await asyncio.to_thread(self.mem0_memory.add, text=text, user_id=user_id, agent_id=agent_id)
#             logger.debug(f"Saved interaction to Mem0 for user {user_id}, agent {agent_id}.")
#         except Exception as e:
#             logger.error(f"Could not save interaction to Mem0: {e}", exc_info=self.verbose)

#     # --- Tool Definition Wrappers (Async where possible) ---
#     # NOTE: The actual tool functions called by the AgentExecutor should ideally be async
#     # If the underlying agent logic (e.g., SQLAgent.run) isn't async, the executor
#     # might run it in a thread pool, or we might need to wrap sync calls here.
#     # For simplicity here, we assume the executor handles sync tool calls appropriately,
#     # but we define the wrapper methods called by _get_tools as potentially async.

#     async def _run_modeling_tool_async(self, data_source: str, request: str) -> str:
#         if not self.modeling_agent_instance: return "Modeling Agent not initialized."
#         try:
#             # Assuming run_modeling_task is synchronous
#             return await asyncio.to_thread(self.modeling_agent_instance.run_modeling_task, data_source, request)
#         except Exception as e: logger.error(f"Modeling tool error: {e}", exc_info=self.verbose); return f"Modeling tool error: {e}"

#     async def _run_visualization_tool_async(self, data_source: str, request: str, output_filename: str) -> str:
#         if not self.visualization_agent_instance: return "Visualization Agent not initialized."
#         try:
#             # Assuming generate_plot is synchronous
#             return await asyncio.to_thread(self.visualization_agent_instance.generate_plot, data_source, request, output_filename)
#         except Exception as e: logger.error(f"Visualization tool error: {e}", exc_info=self.verbose); return f"Visualization tool error: {e}"

#     async def _run_sql_query_tool_async(self, natural_language_query: str) -> str:
#         if not self.sql_agent_instance: return "SQL Agent not initialized."
#         try:
#             # Assuming SQLAgent.run is synchronous
#             response = await asyncio.to_thread(self.sql_agent_instance.run, natural_language_query)
#             if response.get("error"): return f"SQL Error: {response['error']}\nSQL: {response.get('generated_sql', 'N/A')}"
#             return f"Result:\n{response.get('result', 'N/A')}\n\nSQL:\n```sql\n{response.get('generated_sql', 'N/A')}\n```"
#         except Exception as e: logger.error(f"SQL tool error: {e}", exc_info=self.verbose); return f"SQL tool error: {e}"

#     async def _run_forecasting_tool_async(self, data_source: str, request: str) -> str:
#         if not self.forecasting_agent_instance: return "Forecasting Agent not initialized."
#         try:
#             # Assuming run_forecasting_task is synchronous
#             return await asyncio.to_thread(self.forecasting_agent_instance.run_forecasting_task, data_source, request)
#         except Exception as e: logger.error(f"Forecasting tool error: {e}", exc_info=self.verbose); return f"Forecasting tool error: {e}"

#     async def _add_knowledge_tool_async(self, source_path: str) -> str:
#         logger.info(f"--- Knowledge Addition Tool ---")
#         logger.info(f"Source path: {source_path}")
#         try:
#             # load_and_split_documents is likely CPU bound / sync I/O
#             documents = await asyncio.to_thread(load_and_split_documents, source_path, self.workspace_dir)
#             if not documents:
#                 return f"Failed to load or split documents from '{source_path}'. No documents were added."

#             # Use the async helper to add to Chroma/Mem0
#             await self._add_documents_to_chroma_async(documents)
#             return f"Successfully processed and added knowledge from '{source_path}' to the knowledge base."
#         except Exception as e:
#             logger.error(f"Error in knowledge addition tool: {e}", exc_info=self.verbose)
#             return f"An error occurred while adding knowledge from '{source_path}': {e}"

#     def _get_tools(self) -> List[BaseTool]:
#         """Gets all tools available to the Supervisor Agent."""
#         all_tools = []
#         # File Management Tools (Sync by default in LangChain toolkit)
#         try:
#             # Note: File operations are blocking. AgentExecutor will likely run these in threads.
#             fm_toolkit = FileManagementToolkit(root_dir=self.workspace_dir)
#             all_tools.extend(fm_toolkit.get_tools())
#             logger.debug(f"Loaded File Tools: {[t.name for t in fm_toolkit.get_tools()]}")
#         except Exception as e: logger.warning(f"File tools initialization failed: {e}", exc_info=self.verbose)

#         # Wrap async methods for tools
#         if self.sql_agent_instance:
#             all_tools.append(tool(name="query_sql_database", func=self._run_sql_query_tool_async, description="Query SQL DB with natural language.", coroutine=self._run_sql_query_tool_async))
#         if self.visualization_agent_instance:
#              all_tools.append(tool(name="create_visualization", func=self._run_visualization_tool_async, description=f"Create plot from data (path/CSV string in '{self.workspace_dir}'), save as image. Args: data_source, request, output_filename.", coroutine=self._run_visualization_tool_async))
#         if self.modeling_agent_instance:
#              all_tools.append(tool(name="run_modeling_task", func=self._run_modeling_tool_async, description=f"Train/evaluate ML model (classification/regression) from data (path/CSV string in '{self.workspace_dir}'). Request MUST specify target column. Returns metrics & optional saved model path. Args: data_source, request.", coroutine=self._run_modeling_tool_async))
#         if self.forecasting_agent_instance:
#              all_tools.append(tool(name="run_time_series_forecast", func=self._run_forecasting_tool_async, description=f"Generate time series forecast using SARIMAX. Args: data_source (path/CSV string in '{self.workspace_dir}'), request (natural language, MUST specify target col, time col, horizon; optionally freq, orders). Returns metrics, forecast summary, optional saved paths.", coroutine=self._run_forecasting_tool_async))

#         all_tools.append(tool(name="add_knowledge_base_document", func=self._add_knowledge_tool_async, description=f"Add knowledge from a document or directory to the assistant's knowledge base for later retrieval. Args: source_path (path to file/dir in workspace '{self.workspace_dir}').", coroutine=self._add_knowledge_tool_async))

#         logger.info(f"Loaded Tools: {[t.name for t in all_tools]}")
#         if not all_tools: logger.warning("No tools were loaded for Supervisor.")
#         return all_tools

#     def _create_agent_executor(self) -> Optional[AgentExecutor]:
#         """Creates the LangChain Agent Executor."""
#         agent_type = AgentType.OPENAI_FUNCTIONS # Requires LLM support for function calling
#         logger.info(f"Creating Supervisor Agent Executor ({agent_type})")
#         if not self.tools:
#             logger.error("Cannot create agent executor with no tools.")
#             return None

#         # Define the core prompt
#         system_prompt = """You are a helpful and comprehensive assistant.
# You have access to tools for database queries, visualization, modeling, forecasting, file management, and knowledge management.
# You also use an internal knowledge base and conversation memory.
# Prioritize using tools for specific tasks (SQL, plotting, modeling, forecasting, files).
# Use your knowledge base and memory for general questions or information retrieval.
# Summarize tool results clearly. Indicate when using knowledge/memory. State if you cannot fulfill a request."""

#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", system_prompt),
#                 MessagesPlaceholder("chat_history", optional=True),
#                 ("human", "{input}"),
#                 MessagesPlaceholder("agent_scratchpad"),
#             ]
#         )

#         try:
#             # Ensure the agent can handle async tools if specified with coroutine=...
#             agent_executor = initialize_agent(
#                 tools=self.tools,
#                 llm=self.llm,
#                 agent=agent_type,
#                 verbose=self.verbose,
#                 handle_parsing_errors="Check your output format and try again!",
#                 max_iterations=15,
#                 early_stopping_method="generate",
#                 # Ensure the executor knows tools can be async
#                 # (This might be implicit based on tool definition with `coroutine=`)
#             )
#             logger.debug("Supervisor agent executor created successfully.")
#             return agent_executor
#         except Exception as e:
#             logger.error(f"Error initializing supervisor agent executor: {e}", exc_info=self.verbose)
#             return None

#     # --- Interactive Chat Method (Async) ---
#     async def arun_interactive(self, user_input: str, user_id: str = "default_user") -> Dict[str, Any]:
#         """
#         Handles interactive chat requests asynchronously, including RAG and Memory.
#         """
#         logger.info(f"--- Running Supervisor Agent (Interactive Async) --- User ({user_id})")
#         logger.debug(f"User Input: {user_input}")
#         if not self.agent_executor:
#             logger.error("Agent executor not initialized.")
#             return {"error": "Supervisor agent executor not initialized."}

#         agent_response = {}
#         final_answer = "An error occurred during processing."
#         run_config = RunnableConfig(configurable={"user_id": user_id}) # Example config

#         try:
#             # 1. Retrieve context and memory concurrently
#             context_task = asyncio.create_task(self._retrieve_relevant_context_async(user_input))
#             memory_task = asyncio.create_task(self._retrieve_relevant_memories_async(user_input, user_id))

#             # 2. Run the main agent executor (potentially using tools)
#             # Use ainvoke for async execution
#             agent_response_task = asyncio.create_task(
#                 self.agent_executor.ainvoke({"input": user_input}, config=run_config)
#             )

#             # Await all concurrent tasks
#             agent_response = await agent_response_task
#             initial_output = agent_response.get("output", "Agent did not produce a direct output.")
#             logger.debug(f"Initial Agent Output: {initial_output[:150]}...")

#             retrieved_context_chunks = await context_task
#             retrieved_context = "\n\n".join(retrieved_context_chunks) if retrieved_context_chunks else "No relevant context found in knowledge base."

#             relevant_memories = await memory_task
#             mem0_context = ""
#             if relevant_memories:
#                 mem0_context = "\n\nRelevant past interactions:\n" + "\n".join([f"- {m['text']}" for m in relevant_memories])

#             # 3. Synthesize Final Answer using LLM with all context
#             synthesis_prompt = ChatPromptTemplate.from_messages([
#                 SystemMessage(content="Synthesize the final user response based on the query, initial assistant output, retrieved context, and past interactions. Prioritize context, summarize tool use, incorporate memory. Be concise."),
#                 HumanMessage(content=f"User Query:\n{user_input}\n\nInitial Assistant Output:\n{initial_output}\n\nRetrieved Context:\n{retrieved_context}\n\nRelevant Past Interactions:\n{mem0_context}\n\nSynthesized Final Answer:")
#             ])
#             synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
#             logger.debug("Synthesizing final answer...")
#             # Use ainvoke for async LLM call
#             final_answer = await synthesis_chain.ainvoke({})
#             logger.debug(f"Synthesized Answer: {final_answer[:150]}...")

#             # 4. Add final interaction to Mem0 (async)
#             await self._add_memory_async(
#                 text=f"User: {user_input}\nAssistant: {final_answer}",
#                 user_id=user_id,
#                 agent_id="supervisor_final"
#             )

#             agent_response['final_output'] = final_answer

#         except Exception as e:
#             logger.error(f"Error during Supervisor interactive execution or RAG synthesis: {e}", exc_info=self.verbose)
#             final_answer = f"An error occurred: {e}"
#             agent_response['error'] = str(e)
#             agent_response['final_output'] = final_answer

#         logger.info(f"--- Supervisor Agent Finished (Interactive Async) ---")
#         return agent_response

#     # --- Autonomous Task Execution (Async) ---
#     async def _arun_autonomous_task(self, task_prompt: str, task_id: str = "autonomous_task") -> Dict[str, Any]:
#         """
#         Executes a predefined task asynchronously using the agent executor.
#         Bypasses RAG/Memory synthesis, logs output. For scheduled/event tasks.
#         """
#         logger.info(f"--- Running Autonomous Task ({task_id}) ---")
#         logger.info(f"Task Prompt: {task_prompt}")
#         if not self.agent_executor:
#             logger.error(f"Cannot run task {task_id}: Agent executor not initialized.")
#             return {"error": "Supervisor agent executor not initialized."}

#         response = {}
#         output = "Autonomous task did not produce output."
#         try:
#             # Use ainvoke for async execution
#             response = await self.agent_executor.ainvoke({"input": task_prompt})
#             output = response.get("output", output)
#             logger.info(f"Autonomous Task ({task_id}) Output: {output[:200]}...")

#             # --- Log Output ---
#             log_file_path = os.path.join(self.workspace_dir, "autonomous_logs", f"{task_id}.log")
#             os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
#             async with asyncio.Lock(): # Basic lock for async file write safety
#                  with open(log_file_path, "a", encoding='utf-8') as log_file:
#                     timestamp = datetime.now().isoformat()
#                     log_file.write(f"--- {timestamp} ---\n")
#                     log_file.write(f"Task: {task_prompt}\n")
#                     log_file.write(f"Output:\n{output}\n\n")
#             logger.debug(f"Autonomous task ({task_id}) output logged to: {log_file_path}")

#             # --- Add to Mem0 ---
#             await self._add_memory_async(
#                 text=f"Autonomous Task: {task_prompt}\nResult: {output}",
#                 user_id="system_autonomous", # Specific user ID for autonomous tasks
#                 agent_id=task_id
#             )
#             return response

#         except Exception as e:
#             logger.error(f"Error during autonomous task execution ({task_id}): {e}", exc_info=self.verbose)
#             # Log error to a separate file
#             error_log_path = os.path.join(self.workspace_dir, "autonomous_logs", f"{task_id}_error.log")
#             os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
#             async with asyncio.Lock():
#                  with open(error_log_path, "a", encoding='utf-8') as log_file:
#                      timestamp = datetime.now().isoformat()
#                      log_file.write(f"--- {timestamp} ---\n")
#                      log_file.write(f"Task: {task_prompt}\n")
#                      log_file.write(f"Error: {e}\n{traceback.format_exc()}\n\n")
#             logger.error(f"Autonomous task ({task_id}) error logged to: {error_log_path}")
#             return {"error": str(e)}

#     # --- Event Handling ---
#     async def enqueue_event(self, event_type: str, event_data: Dict[str, Any]):
#         """Adds an event to the processing queue."""
#         logger.info(f"Event received: Type='{event_type}', Data='{event_data}'. Enqueuing...")
#         await self.event_queue.put((event_type, event_data))

#     async def _process_event_queue(self):
#         """Continuously processes events from the queue."""
#         logger.info("Starting event queue processor...")
#         while True:
#             try:
#                 event_type, event_data = await self.event_queue.get()
#                 logger.info(f"Processing event: Type='{event_type}'")
#                 task_prompt = None
#                 task_id_base = f"event_{event_type}_{int(time.time())}"

#                 # --- Logic to convert event to task prompt ---
#                 if event_type == "new_file":
#                     file_path = event_data.get("file_path")
#                     if file_path:
#                         # Make path relative to workspace if possible for the prompt
#                         rel_path = os.path.relpath(file_path, self.workspace_dir) if file_path.startswith(self.workspace_dir) else file_path
#                         filename = os.path.basename(rel_path)
#                         task_prompt = f"A new file '{filename}' appeared at '{rel_path}'. Analyze its content and provide a summary."
#                         task_id_base = f"event_newfile_{filename.split('.')[0]}"
#                     else: logger.warning("New file event missing 'file_path'.")

#                 elif event_type == "api_trigger":
#                     payload = event_data.get("payload")
#                     endpoint = event_data.get("endpoint")
#                     task_prompt = f"Received API trigger on endpoint '{endpoint}' with payload: {json.dumps(payload)}. Perform the necessary action based on this data."
#                     task_id_base = f"event_api_{endpoint}"

#                 # Add more event types here...

#                 # --- Execute task ---
#                 if task_prompt:
#                     await self._arun_autonomous_task(task_prompt, task_id=task_id_base)
#                 else:
#                     logger.warning(f"No task prompt generated for event type '{event_type}'. Skipping.")

#                 self.event_queue.task_done() # Mark task as completed

#             except asyncio.CancelledError:
#                  logger.info("Event queue processor cancelled.")
#                  break
#             except Exception as e:
#                  logger.error(f"Error in event queue processor: {e}", exc_info=self.verbose)
#                  # Avoid crashing the loop, maybe add a delay before retrying?
#                  await asyncio.sleep(5)

#     # --- Scheduling ---
#     def add_scheduled_task(self, task_func, trigger, task_id: str, task_prompt: str, **trigger_args):
#         """Adds a job to the APScheduler."""
#         if not callable(task_func):
#              logger.error(f"Cannot schedule task '{task_id}': task_func is not callable.")
#              return

#         logger.info(f"Adding scheduled task '{task_id}' with trigger '{trigger}'")
#         try:
#             self.scheduler.add_job(
#                 task_func,
#                 trigger=trigger,
#                 args=[task_prompt, task_id], # Arguments for _arun_autonomous_task
#                 id=task_id, # Unique ID for the job
#                 replace_existing=True, # Replace if job with same ID exists
#                 **trigger_args
#             )
#         except Exception as e:
#             logger.error(f"Failed to add scheduled task '{task_id}': {e}", exc_info=self.verbose)

#     # --- Goal Management (Conceptual Placeholders) ---
#     async def start_goal(self, goal_description: str, goal_type: str = "default") -> str:
#         """
#         Initiates a long-running, goal-oriented task using LangGraph.
#         (This is a conceptual implementation)
#         """
#         goal_id = f"goal_{goal_type}_{uuid.uuid4()}"
#         logger.info(f"Initiating goal ({goal_id}): {goal_description}")

#         # 1. Select and build the appropriate LangGraph app based on goal_type
#         graph_app = None
#         initial_state = {}
#         if goal_type == "website_monitor":
#             # Example: Load and compile graph defined elsewhere
#             # graph_builder = build_monitor_graph() # Assume this returns compiled app
#             # graph_app = graph_builder.compile(checkpointer=SqliteSaver.from_conn_string(":memory:")) # Example persistence
#             # initial_state = {"target_url": "...", "max_checks": ...} # Extract from goal_description
#             logger.warning("Website monitor goal type is conceptual. Graph not implemented.")
#             return f"Error: Goal type '{goal_type}' graph not implemented."
#         # Add other goal types here...
#         else:
#             logger.error(f"Unknown goal type: {goal_type}")
#             return f"Error: Unknown goal type '{goal_type}'"

#         # 2. Start the graph execution asynchronously
#         if graph_app:
#             try:
#                 # Run the graph in the background
#                 # Use astream for continuous updates or ainvoke for final result
#                 graph_task = asyncio.create_task(graph_app.ainvoke(initial_state, config={"configurable": {"thread_id": goal_id}}))
#                 self.running_goals[goal_id] = graph_task
#                 logger.info(f"LangGraph task started for goal {goal_id}.")
#                 return goal_id # Return the ID to track the goal
#             except Exception as e:
#                  logger.error(f"Failed to start LangGraph task for goal {goal_id}: {e}", exc_info=self.verbose)
#                  return f"Error starting goal: {e}"
#         else:
#              # Error already logged
#              return f"Error: Could not create graph for goal type '{goal_type}'."


#     async def get_goal_status(self, goal_id: str) -> Dict[str, Any]:
#         """Checks the status of a running LangGraph goal."""
#         if goal_id not in self.running_goals:
#             return {"status": "not_found", "message": f"Goal ID {goal_id} not found."}

#         task = self.running_goals[goal_id]
#         if task.done():
#             try:
#                 result = task.result()
#                 # Clean up completed task
#                 del self.running_goals[goal_id]
#                 logger.info(f"Goal {goal_id} completed.")
#                 # Persist final state? Depends on graph checkpointer setup
#                 return {"status": "completed", "result": result} # Or final state from checkpointer
#             except Exception as e:
#                 logger.error(f"Goal {goal_id} finished with error: {e}", exc_info=self.verbose)
#                 del self.running_goals[goal_id] # Clean up failed task
#                 return {"status": "error", "error": str(e)}
#         else:
#             # For ongoing tasks, might need to query the graph's checkpointer for current state
#             logger.debug(f"Goal {goal_id} is still running.")
#             # state = await graph_app.aget_state(...) # Requires checkpointer access
#             return {"status": "running"} # "current_step": state.get(...)}


#     # --- Main Lifecycle Methods ---
#     async def start(self):
#         """Starts the scheduler and event queue processor."""
#         logger.info("Starting Supervisor Agent background tasks...")
#         # Start the scheduler if jobs are added
#         if self.scheduler.get_jobs():
#             self.scheduler.start()
#             logger.info("Scheduler started.")
#         else:
#             logger.info("Scheduler not started (no jobs added).")

#         # Start the event queue processor as a background task
#         self.event_processor_task = asyncio.create_task(self._process_event_queue())
#         logger.info("Event queue processor task created.")

#     async def stop(self):
#         """Gracefully stops the scheduler and event queue processor."""
#         logger.info("Stopping Supervisor Agent background tasks...")
#         # Shutdown scheduler
#         if self.scheduler.running:
#             self.scheduler.shutdown()
#             logger.info("Scheduler shut down.")

#         # Cancel event processor task
#         if hasattr(self, 'event_processor_task') and not self.event_processor_task.done():
#             self.event_processor_task.cancel()
#             try:
#                 await self.event_processor_task # Wait for cancellation
#             except asyncio.CancelledError:
#                 logger.info("Event queue processor task cancelled successfully.")
#         # Cancel any running goal tasks
#         logger.info(f"Cancelling {len(self.running_goals)} running goal tasks...")
#         for goal_id, task in list(self.running_goals.items()):
#              if not task.done():
#                  task.cancel()
#                  try: await task
#                  except asyncio.CancelledError: logger.debug(f"Goal task {goal_id} cancelled.")
#                  except Exception as e: logger.warning(f"Error during goal task {goal_id} cancellation: {e}")
#              del self.running_goals[goal_id] # Remove from tracking
#         logger.info("Supervisor Agent stopped.")

