# handlers/goal_handler.py

import logging
import asyncio
import uuid
import json
from typing import Dict, Any, Optional, Tuple

# --- LangGraph & Checkpointing ---
from langgraph.graph import StateGraph, END, CompiledGraph
# Use MemorySaver for simplicity, switch to SqliteSaver for persistence
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.sqlite import SqliteSaver # For persistent state

# --- LangChain ---
from langchain_core.prompts import ChatPromptTemplate, SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

# --- Project Imports ---
from ..core.core_services import CoreAgentServices
from ..utils import config # For potential config values like checkpoint DB path

# --- Placeholder for Graph Definitions ---
try:
    # --- UPDATE THIS LINE ---
    from ..graphs.website_monitor import get_website_monitor_app
    # --- END UPDATE ---
    # Add imports for other graph types here
    # from ..graphs.data_pipeline import get_data_pipeline_app
    pass
except ImportError as e:
    # Log the specific import error
    logging.warning(f"Could not import specific graph definitions: {e}. Goal execution will fail for those types.")
    # Define dummy functions to avoid NameErrors if needed during development
    def get_website_monitor_app(checkpointer): return None
    # def get_data_pipeline_app(checkpointer): return None



class GoalHandler:
    """
    Manages long-running, stateful, goal-oriented tasks using LangGraph.
    """
    def __init__(self, services: CoreAgentServices):
        """
        Initializes the GoalHandler.

        Args:
            services: The shared CoreAgentServices instance.
        """
        self.services = services
        self.logger = logging.getLogger(__name__)

        # --- State Management ---
        # Use MemorySaver for simplicity (state lost on restart)
        self.checkpointer = MemorySaver()
        # For persistence, use SqliteSaver:
        # checkpoint_db_path = os.path.join(config.WORKSPACE_DIR, "goal_checkpoints.sqlite")
        # self.checkpointer = SqliteSaver.from_conn_string(checkpoint_db_path)
        self.logger.info(f"Using checkpointer: {self.checkpointer.__class__.__name__}")

        # --- Tracking Active Goals ---
        # Stores {goal_id: asyncio.Task} for running graph invocations
        self.running_goal_tasks: Dict[str, asyncio.Task] = {}
        # Stores {goal_id: CompiledGraph} to interact with specific graph instances later
        self.goal_apps: Dict[str, CompiledGraph] = {}

        self.logger.info("GoalHandler initialized.")

    async def _parse_goal_to_state(self, goal_description: str, goal_type: str) -> Optional[Dict[str, Any]]:
        """Uses LLM to parse natural language goal into the initial state dictionary for a specific graph type."""
        self.logger.debug(f"Parsing goal description for type '{goal_type}': '{goal_description[:100]}...'")
        parser = JsonOutputParser()
        system_prompt = f"""
You are an expert at understanding goal descriptions and extracting the necessary initial parameters for a specific type of automated task graph (goal_type='{goal_type}').
Analyze the user's goal description and return a JSON object containing the required initial state variables for that graph type.

Known Goal Types & Required State Variables:
- 'website_monitor': Requires `target_url` (string), `max_checks` (integer, default 3 if not specified).
- 'data_processing_pipeline': Requires `input_file` (string, path relative to workspace), `output_file` (string, path relative to workspace), `steps` (list of strings describing processing steps, e.g., ["clean_data", "calculate_averages"]).
# Add more goal types and their required state variables here...

If the goal description clearly matches a known type and provides the necessary information, extract it into the JSON.
If the type is unknown or required information is missing, return an error structure: {{"error": "Reason..."}}.
"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Parse the following goal description:\n\n{goal_description}")
        ])
        chain = prompt | self.services.llm | parser

        try:
            parsed_state = await chain.ainvoke({}) # Assuming LLM supports async
            self.logger.debug(f"LLM parsed initial state: {parsed_state}")
            if not isinstance(parsed_state, dict):
                 self.logger.warning("LLM state parsing did not return a dict.")
                 return {"error": "LLM parsing failed to return a dictionary."}
            if "error" in parsed_state:
                 self.logger.warning(f"LLM parsing returned error: {parsed_state['error']}")
                 return parsed_state # Propagate error

            # --- Add specific validation/defaults based on goal_type ---
            if goal_type == "website_monitor":
                if "target_url" not in parsed_state: return {"error": "Missing 'target_url' for website_monitor goal."}
                if "max_checks" not in parsed_state: parsed_state["max_checks"] = 3 # Default
                elif not isinstance(parsed_state["max_checks"], int) or parsed_state["max_checks"] <= 0:
                     return {"error": "'max_checks' must be a positive integer."}
            elif goal_type == "data_processing_pipeline":
                 if not all(k in parsed_state for k in ["input_file", "output_file", "steps"]):
                      return {"error": "Missing 'input_file', 'output_file', or 'steps' for data_processing_pipeline goal."}
                 if not isinstance(parsed_state["steps"], list): return {"error": "'steps' must be a list."}
            # Add validation for other goal types...

            return parsed_state
        except Exception as e:
            self.logger.error(f"Error parsing goal description with LLM: {e}", exc_info=self.services.verbose)
            return {"error": f"LLM parsing failed: {e}"}

    def _get_graph_app(self, goal_type: str) -> Optional[CompiledGraph]:
        """Selects and returns the compiled LangGraph app for the given goal type."""
        self.logger.debug(f"Getting graph application for goal type: {goal_type}")
        app = None
        try:
            if goal_type == "website_monitor":
                # --- UPDATE THIS LINE ---
                # Assumes get_website_monitor_app takes the checkpointer
                app = get_website_monitor_app(checkpointer=self.checkpointer)
                # --- END UPDATE ---
            elif goal_type == "data_processing_pipeline":
                 # app = get_data_pipeline_app(checkpointer=self.checkpointer)
                 self.logger.warning(f"Graph definition for '{goal_type}' not implemented/imported.")
                 pass # Add other graph types here
            else:
                self.logger.error(f"Unknown goal type specified: {goal_type}")
                return None

            if app is None:
                 self.logger.error(f"Failed to get or build graph application for type '{goal_type}'. Check graph definition function.")
                 return None

            self.logger.debug(f"Successfully obtained graph application for '{goal_type}'.")
            return app
        except Exception as e:
             self.logger.error(f"Error getting graph application for type '{goal_type}': {e}", exc_info=self.services.verbose)
             return None


    async def start_goal(self, goal_description: str, goal_type: str) -> Dict[str, Any]:
        """
        Parses a goal description, selects the appropriate graph,
        and starts its execution asynchronously.

        Args:
            goal_description: Natural language description of the goal.
            goal_type: Identifier for the type of graph to use (e.g., 'website_monitor').

        Returns:
            A dictionary containing 'goal_id' on success, or 'error' on failure.
        """
        goal_id = f"goal_{goal_type}_{uuid.uuid4()}"
        self.logger.info(f"Attempting to start goal ({goal_id}) of type '{goal_type}': {goal_description[:100]}...")

        # 1. Parse description into initial state
        initial_state = await self._parse_goal_to_state(goal_description, goal_type)
        if not initial_state or "error" in initial_state:
            error_msg = f"Failed to parse goal description: {initial_state.get('error', 'Unknown parsing error.') if isinstance(initial_state, dict) else 'Parsing failed.'}"
            self.logger.error(error_msg)
            return {"error": error_msg}

        # 2. Get the compiled graph application
        graph_app = self._get_graph_app(goal_type)
        if not graph_app:
            error_msg = f"Could not find or build graph application for goal type '{goal_type}'."
            self.logger.error(error_msg)
            return {"error": error_msg}

        # 3. Start graph execution asynchronously
        try:
            # The config associates this run with the goal_id for the checkpointer
            config_for_run = {"configurable": {"thread_id": goal_id}}
            self.logger.debug(f"Invoking graph for goal {goal_id} with initial state: {initial_state} and config: {config_for_run}")

            # Run the graph in the background using ainvoke
            graph_task = asyncio.create_task(graph_app.ainvoke(initial_state, config=config_for_run))

            # Store task and app instance for status checking / cancellation
            self.running_goal_tasks[goal_id] = graph_task
            self.goal_apps[goal_id] = graph_app # Store the app instance associated with the goal

            self.logger.info(f"Successfully started background task for goal {goal_id}.")
            return {"goal_id": goal_id, "status": "started"}

        except Exception as e:
            self.logger.error(f"Failed to invoke or start LangGraph task for goal {goal_id}: {e}", exc_info=self.services.verbose)
            return {"error": f"Error starting graph execution: {e}"}

    async def get_goal_status(self, goal_id: str) -> Dict[str, Any]:
        """
        Checks the status and current/final state of a running or completed goal.

        Args:
            goal_id: The unique ID of the goal to check.

        Returns:
            A dictionary containing the goal's status ('running', 'completed', 'error', 'not_found')
            and potentially the current/final state or error message.
        """
        self.logger.debug(f"Checking status for goal {goal_id}...")

        # Check if we have the app instance for this goal
        graph_app = self.goal_apps.get(goal_id)
        if not graph_app:
             # Maybe it finished and was cleaned up? Check checkpointer directly.
             try:
                 final_state = await asyncio.to_thread(self.checkpointer.get, {"configurable": {"thread_id": goal_id}})
                 if final_state:
                      self.logger.info(f"Goal {goal_id} not actively running but found completed state in checkpointer.")
                      # Determine if it ended normally or with error based on state content (graph specific)
                      # This requires the graph to store error info in its state if it fails
                      is_error = final_state.get("error") is not None # Example check
                      return {"goal_id": goal_id, "status": "error" if is_error else "completed", "state": final_state}
                 else:
                      self.logger.warning(f"Goal ID {goal_id} not found in active tasks or checkpointer.")
                      return {"goal_id": goal_id, "status": "not_found"}
             except Exception as e:
                  self.logger.error(f"Error checking checkpointer for completed goal {goal_id}: {e}", exc_info=self.services.verbose)
                  return {"goal_id": goal_id, "status": "unknown", "error": f"Error checking status: {e}"}


        task = self.running_goal_tasks.get(goal_id)
        config_for_run = {"configurable": {"thread_id": goal_id}}

        if task and not task.done():
            # Task is still running, get current state from checkpointer
            try:
                current_state = await asyncio.to_thread(graph_app.get_state, config=config_for_run)
                self.logger.debug(f"Goal {goal_id} is running. Current state snapshot: {current_state}")
                # Extract relevant info from state (e.g., last message, current step) - depends on graph state definition
                status_detail = {"current_step": "Unknown", "last_message": None} # Example structure
                if hasattr(current_state, 'values') and isinstance(current_state.values, dict):
                     # Try common patterns, adjust based on your actual graph state structure
                     status_detail["last_message"] = current_state.values.get("messages", [{}])[-1] if current_state.values.get("messages") else None
                     # Getting current step name might require inspecting graph internals or storing it in state

                return {"goal_id": goal_id, "status": "running", "state_snapshot": status_detail}
            except Exception as e:
                self.logger.error(f"Error getting current state for running goal {goal_id}: {e}", exc_info=self.services.verbose)
                return {"goal_id": goal_id, "status": "running", "error": f"Could not retrieve current state: {e}"}

        elif task and task.done():
            # Task has finished, get final state and handle result/error
            self.logger.info(f"Goal {goal_id} task has finished. Retrieving final state...")
            final_state = None
            error_message = None
            try:
                # Check if the task itself raised an exception
                task_exception = task.exception()
                if task_exception:
                    raise task_exception # Raise it to be caught below

                # If no task exception, get final state from checkpointer
                final_state = await asyncio.to_thread(graph_app.get_state, config=config_for_run)
                self.logger.info(f"Final state for goal {goal_id}: {final_state}")
                status = "completed"
                # Check if the graph itself recorded an error in its final state
                if hasattr(final_state, 'values') and isinstance(final_state.values, dict) and final_state.values.get("error"):
                     status = "error"
                     error_message = final_state.values["error"]

            except Exception as e:
                self.logger.error(f"Goal {goal_id} task finished with error or failed state retrieval: {e}", exc_info=self.services.verbose)
                status = "error"
                error_message = str(e)
            finally:
                # Clean up completed/failed task
                if goal_id in self.running_goal_tasks: del self.running_goal_tasks[goal_id]
                if goal_id in self.goal_apps: del self.goal_apps[goal_id]
                self.logger.debug(f"Cleaned up task references for goal {goal_id}.")

            return {"goal_id": goal_id, "status": status, "final_state": final_state, "error": error_message}

        else:
             # Should have been caught by the initial check, but as a fallback
             self.logger.warning(f"Goal ID {goal_id} not found in active tasks map.")
             return {"goal_id": goal_id, "status": "not_found"}


    async def cancel_goal(self, goal_id: str) -> Dict[str, Any]:
        """
        Attempts to cancel a running goal task.

        Args:
            goal_id: The unique ID of the goal to cancel.

        Returns:
            A dictionary indicating the result ('cancelled', 'not_running', 'error').
        """
        self.logger.info(f"Attempting to cancel goal {goal_id}...")
        task = self.running_goal_tasks.get(goal_id)

        if task and not task.done():
            try:
                task.cancel()
                await asyncio.sleep(0.1) # Give cancellation a moment to propagate
                if task.cancelled():
                     self.logger.info(f"Goal {goal_id} task cancelled successfully.")
                     status = "cancelled"
                else:
                     # This might happen if cancellation is blocked internally
                     self.logger.warning(f"Sent cancel request for goal {goal_id}, but task state is not 'cancelled'. It might finish or error.")
                     status = "cancel_requested" # Indicate cancellation was attempted

                # Clean up references regardless of exact cancelled state
                if goal_id in self.running_goal_tasks: del self.running_goal_tasks[goal_id]
                if goal_id in self.goal_apps: del self.goal_apps[goal_id]
                return {"goal_id": goal_id, "status": status}

            except Exception as e:
                self.logger.error(f"Error during cancellation of goal {goal_id}: {e}", exc_info=self.services.verbose)
                return {"goal_id": goal_id, "status": "error", "error": f"Error during cancellation: {e}"}
        elif task and task.done():
            self.logger.warning(f"Cannot cancel goal {goal_id}: Task already completed.")
            # Clean up if somehow missed before
            if goal_id in self.running_goal_tasks: del self.running_goal_tasks[goal_id]
            if goal_id in self.goal_apps: del self.goal_apps[goal_id]
            return {"goal_id": goal_id, "status": "already_completed"}
        else:
            self.logger.warning(f"Cannot cancel goal {goal_id}: Goal ID not found or not running.")
            return {"goal_id": goal_id, "status": "not_found"}

