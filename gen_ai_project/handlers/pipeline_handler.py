# handlers/pipeline_handler.py

import logging
import asyncio
import uuid
import json
import time
import os # Added os import
import yaml # Added yaml import
from typing import Dict, Any, Optional, List, Callable

# --- Project Imports ---
from ..core.core_services import CoreAgentServices
# Import config to get CONFIG_DIR
from ..utils import config

class PipelineHandler:
    """
    Manages the execution of predefined, sequential workflows (pipelines).
    Pipelines consist of steps that typically call tools via CoreAgentServices.
    """
    def __init__(self, services: CoreAgentServices):
        """
        Initializes the PipelineHandler.

        Args:
            services: The shared CoreAgentServices instance.
        """
        self.services = services
        self.logger = logging.getLogger(__name__)

        # --- Pipeline Definitions ---
        # Load definitions from external file
        self._pipelines: Dict[str, List[Dict[str, Any]]] = self._load_pipelines()

        # --- Tracking Active Pipeline Runs ---
        # Stores {run_id: asyncio.Task} for running pipeline executions
        self.running_pipelines: Dict[str, asyncio.Task] = {}
        # Stores {run_id: {"status": "...", "result": ..., "error": ...}} for completed/failed runs
        self.pipeline_results: Dict[str, Dict[str, Any]] = {}
        # TODO: Consider limiting the size of pipeline_results or using a persistent store

        self.logger.info("PipelineHandler initialized.")

    def _load_pipelines(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loads pipeline definitions from the pipelines.yaml file.
        """
        self.logger.info("Loading pipeline definitions from YAML file...")
        definitions = {}
        # Construct the path to the YAML file using CONFIG_DIR from utils.config
        pipelines_file_path = os.path.join(config.CONFIG_DIR, "pipelines.yaml")

        try:
            if not os.path.exists(pipelines_file_path):
                 self.logger.warning(f"Pipelines configuration file not found at: {pipelines_file_path}. No pipelines loaded.")
                 return {}

            with open(pipelines_file_path, 'r', encoding='utf-8') as f:
                definitions = yaml.safe_load(f)
                if not isinstance(definitions, dict):
                     self.logger.error(f"Failed to load pipelines: YAML content in '{pipelines_file_path}' is not a valid dictionary (mapping).")
                     return {}

            # Basic validation (optional but recommended)
            valid_definitions = {}
            for pipe_id, steps in definitions.items():
                if not isinstance(steps, list):
                    self.logger.warning(f"Pipeline '{pipe_id}' definition is not a list of steps. Skipping.")
                    continue
                is_valid_pipeline = True
                for i, step in enumerate(steps):
                    if not isinstance(step, dict) or "tool" not in step:
                         self.logger.warning(f"Step {i+1} in pipeline '{pipe_id}' is invalid (not a dict or missing 'tool'). Skipping pipeline.")
                         is_valid_pipeline = False
                         break # Stop validating this pipeline
                if is_valid_pipeline:
                     valid_definitions[pipe_id] = steps

            loaded_count = len(valid_definitions)
            self.logger.info(f"Successfully loaded and validated {loaded_count} pipeline definitions from {pipelines_file_path}.")
            return valid_definitions

        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing pipelines YAML file '{pipelines_file_path}': {e}", exc_info=True)
            return {}
        except FileNotFoundError:
             # This case is handled by the os.path.exists check above, but keep for safety
             self.logger.error(f"Pipelines configuration file not found at: {pipelines_file_path}")
             return {}
        except Exception as e:
            self.logger.error(f"An unexpected error occurred loading pipelines from '{pipelines_file_path}': {e}", exc_info=True)
            return {}

    def _resolve_params(self, params_template: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves parameter values using placeholders from the context.
        Simple implementation using string replacement.
        Placeholders: {{initial_context.key}}, {{context.key}} (or {{context.step_id_output_key}})
        """
        resolved_params = {}
        for key, value_template in params_template.items():
            if isinstance(value_template, str):
                try:
                    # Create a combined context for formatting
                    # Use a flat dict for easier replacement logic
                    flat_context = {}
                    flat_context.update({f"initial_context.{k}": v for k, v in context.get("initial_context", {}).items()})
                    # Add step outputs directly under context.key
                    flat_context.update({f"context.{k}": v for k, v in context.items() if k != "initial_context"})

                    resolved_value = value_template
                    # Iteratively replace placeholders
                    # This is basic; Jinja2 would be more robust for complex logic/filters
                    for placeholder, ctx_value in flat_context.items():
                         resolved_value = resolved_value.replace(f"{{{{{placeholder}}}}}", str(ctx_value))

                    # Check if any placeholders remain unresolved (optional warning)
                    if "{{" in resolved_value and "}}" in resolved_value:
                         self.logger.warning(f"Potential unresolved placeholder in param '{key}': {resolved_value}")

                    resolved_params[key] = resolved_value

                except Exception as e:
                    self.logger.warning(f"Failed to resolve parameter template for key '{key}': {value_template}. Error: {e}. Using raw template.")
                    resolved_params[key] = value_template # Fallback to raw template
            else:
                resolved_params[key] = value_template # Pass non-strings directly
        return resolved_params

    async def _execute_pipeline(self, pipeline_id: str, run_id: str, steps: List[Dict[str, Any]], initial_context: Dict[str, Any]):
        """Executes the steps of a single pipeline run."""
        self.logger.info(f"Executing pipeline run '{run_id}' (Pipeline ID: '{pipeline_id}')")
        current_context = {"initial_context": initial_context.copy()} # Start with initial context
        final_status = "completed"
        final_result = None
        error_message = None

        try:
            for i, step in enumerate(steps):
                step_id = step.get("step_id", f"step_{i+1}")
                tool_name = step.get("tool")
                params_template = step.get("params", {})
                output_key = step.get("output_key")
                description = step.get("description", "No description")
                self.logger.info(f"Run '{run_id}', Step '{step_id}': {description} (Tool: {tool_name})")

                if not tool_name:
                    self.logger.warning(f"Run '{run_id}', Step '{step_id}': Skipping step - no tool specified.")
                    continue

                # Resolve parameters using current context
                params = self._resolve_params(params_template, current_context)
                self.logger.debug(f"Run '{run_id}', Step '{step_id}': Resolved Params: {params}")

                # --- Find the tool/method in CoreAgentServices ---
                tool_func: Optional[Callable] = None
                is_async = False

                # 1. Check for async wrappers (convention: _tool_name_async)
                wrapper_name = f"_{tool_name}_async"
                if hasattr(self.services, wrapper_name):
                    tool_func = getattr(self.services, wrapper_name)
                    if callable(tool_func) and asyncio.iscoroutinefunction(tool_func):
                        is_async = True
                        self.logger.debug(f"Found async wrapper: {wrapper_name}")

                # 2. Check for direct methods on CoreAgentServices (sync or async)
                if not tool_func and hasattr(self.services, tool_name):
                     potential_func = getattr(self.services, tool_name)
                     if callable(potential_func):
                          tool_func = potential_func
                          is_async = asyncio.iscoroutinefunction(tool_func)
                          self.logger.debug(f"Found direct method on CoreServices: {tool_name} (async={is_async})")

                # 3. Check for methods on sub-handlers/agents (e.g., knowledge_handler.add_source)
                if not tool_func and '.' in tool_name:
                    parts = tool_name.split('.', 1)
                    handler_attr_name = parts[0]
                    method_name = parts[1]
                    handler_instance = getattr(self.services, handler_attr_name, None)
                    if handler_instance and hasattr(handler_instance, method_name):
                         potential_func = getattr(handler_instance, method_name)
                         if callable(potential_func):
                              tool_func = potential_func
                              is_async = asyncio.iscoroutinefunction(tool_func)
                              self.logger.debug(f"Found method on sub-handler: {tool_name} (async={is_async})")

                # 4. If still not found, raise error
                if not tool_func:
                    raise ValueError(f"Tool '{tool_name}' specified in step '{step_id}' not found or not callable in CoreAgentServices or its direct members/handlers.")

                # --- Execute the tool/method ---
                step_output = None
                try:
                    if is_async:
                        step_output = await tool_func(**params)
                    else:
                        # Run sync function in thread pool
                        step_output = await self.services._run_sync_in_thread(tool_func, **params)

                    self.logger.debug(f"Run '{run_id}', Step '{step_id}': Output: {str(step_output)[:200]}...")

                    # Store output in context if key is specified
                    if output_key:
                        current_context[output_key] = step_output
                        self.logger.debug(f"Run '{run_id}', Step '{step_id}': Stored output under context key '{output_key}'.")
                    else:
                         # Store under a default key if no key specified? Or just log?
                         current_context["_last_step_output"] = step_output

                except Exception as step_err:
                    self.logger.error(f"Run '{run_id}', Step '{step_id}': Execution failed: {step_err}", exc_info=self.services.verbose)
                    raise RuntimeError(f"Pipeline failed at step '{step_id}': {step_err}") from step_err

            # Pipeline completed successfully
            final_result = current_context # Return the final context containing all outputs
            self.logger.info(f"Pipeline run '{run_id}' completed successfully.")

        except Exception as pipeline_err:
            final_status = "error"
            error_message = str(pipeline_err)
            final_result = current_context # Return context up to the point of failure
            self.logger.error(f"Pipeline run '{run_id}' failed: {error_message}")
        finally:
            # Store final status and result
            self.pipeline_results[run_id] = {
                "status": final_status,
                "result": final_result,
                "error": error_message,
                "end_time": time.time(),
                "pipeline_id": pipeline_id # Store original pipeline ID for context
            }
            # Clean up from running tasks
            if run_id in self.running_pipelines:
                del self.running_pipelines[run_id]
            self.logger.debug(f"Stored final result for run '{run_id}' and cleaned up task reference.")


    async def start_pipeline(self, pipeline_id: str, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Starts the execution of a defined pipeline asynchronously.

        Args:
            pipeline_id: The ID of the pipeline definition to run.
            initial_context: Optional dictionary containing initial data/parameters
                             needed by the pipeline (e.g., {"file_path": "data.csv"}).

        Returns:
            A dictionary containing 'run_id' and 'status' on success, or 'error'.
        """
        self.logger.info(f"Request received to start pipeline '{pipeline_id}'")
        if initial_context is None:
            initial_context = {}

        # 1. Find pipeline definition
        pipeline_steps = self._pipelines.get(pipeline_id)
        if not pipeline_steps:
            error_msg = f"Pipeline definition ID '{pipeline_id}' not found or is invalid."
            self.logger.error(error_msg)
            return {"error": error_msg}

        # 2. Generate unique run ID
        run_id = f"pipe_{pipeline_id}_{uuid.uuid4()}"

        # 3. Create and store the execution task
        self.logger.info(f"Creating background task for pipeline run '{run_id}'...")
        pipeline_task = asyncio.create_task(
            self._execute_pipeline(pipeline_id, run_id, pipeline_steps, initial_context)
        )
        self.running_pipelines[run_id] = pipeline_task

        # Store initial status
        self.pipeline_results[run_id] = {
            "status": "running",
            "result": None,
            "error": None,
            "start_time": time.time(),
            "pipeline_id": pipeline_id
        }

        return {"run_id": run_id, "status": "started"}

    def get_pipeline_status(self, run_id: str) -> Dict[str, Any]:
        """
        Checks the status and potentially the result of a pipeline run.

        Args:
            run_id: The unique ID of the pipeline run.

        Returns:
            A dictionary containing the run's status ('running', 'completed', 'error', 'not_found')
            and potentially the result or error message.
        """
        self.logger.debug(f"Checking status for pipeline run '{run_id}'")

        if run_id in self.running_pipelines:
            task = self.running_pipelines[run_id]
            if task.done():
                 self.logger.warning(f"Pipeline run '{run_id}' task is done but still in running_pipelines dict. Checking results.")
                 # Fall through to check pipeline_results
            else:
                 # Return the initial status record while running
                 initial_status = self.pipeline_results.get(run_id, {"status": "running"}) # Should exist
                 return {"run_id": run_id, **initial_status.copy()}

        result_info = self.pipeline_results.get(run_id)
        if result_info:
            # Return a copy to avoid external modification
            return {"run_id": run_id, **result_info.copy()}
        else:
            self.logger.warning(f"Pipeline run ID '{run_id}' not found.")
            return {"run_id": run_id, "status": "not_found"}

    async def cancel_pipeline(self, run_id: str) -> Dict[str, Any]:
        """
        Attempts to cancel a running pipeline task.

        Args:
            run_id: The unique ID of the pipeline run to cancel.

        Returns:
            A dictionary indicating the result ('cancelled', 'not_running', 'error', 'already_completed').
        """
        self.logger.info(f"Attempting to cancel pipeline run '{run_id}'...")
        task = self.running_pipelines.get(run_id)

        if task and not task.done():
            try:
                task.cancel()
                await asyncio.sleep(0.1) # Allow cancellation to process
                status = "cancelled" if task.cancelled() else "cancel_requested"
                self.logger.info(f"Pipeline run '{run_id}' cancellation status: {status}")

                # Update status in results and clean up
                if run_id in self.pipeline_results:
                     self.pipeline_results[run_id]["status"] = status
                     self.pipeline_results[run_id]["error"] = "Pipeline run cancelled by user."
                     self.pipeline_results[run_id]["end_time"] = time.time() # Mark end time
                if run_id in self.running_pipelines: del self.running_pipelines[run_id]
                return {"run_id": run_id, "status": status}

            except Exception as e:
                self.logger.error(f"Error during cancellation of pipeline run '{run_id}': {e}", exc_info=self.services.verbose)
                # Update status even if cancellation had issues
                if run_id in self.pipeline_results:
                     self.pipeline_results[run_id]["status"] = "error"
                     self.pipeline_results[run_id]["error"] = f"Error during cancellation: {e}"
                     self.pipeline_results[run_id]["end_time"] = time.time()
                if run_id in self.running_pipelines: del self.running_pipelines[run_id]
                return {"run_id": run_id, "status": "error", "error": f"Error during cancellation: {e}"}
        elif run_id in self.pipeline_results: # Check if already completed/failed
             status = self.pipeline_results[run_id].get("status", "unknown")
             self.logger.warning(f"Cannot cancel pipeline run '{run_id}': Already finished with status '{status}'.")
             return {"run_id": run_id, "status": f"already_{status}"} # e.g., already_completed
        else:
            self.logger.warning(f"Cannot cancel pipeline run '{run_id}': Run ID not found.")
            return {"run_id": run_id, "status": "not_found"}

    async def stop(self):
        """Cancels all running pipeline tasks on shutdown."""
        self.logger.info("Stopping PipelineHandler: Cancelling all running pipelines...")
        running_ids = list(self.running_pipelines.keys())
        if running_ids:
            self.logger.info(f"Found {len(running_ids)} running pipelines to cancel.")
            await asyncio.gather(*(self.cancel_pipeline(run_id) for run_id in running_ids), return_exceptions=True)
        else:
            self.logger.info("No running pipelines to cancel.")
        self.logger.info("PipelineHandler stopped.")

