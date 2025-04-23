# handlers/event_handler.py

import asyncio
import logging
import json
import time
import os
from typing import Dict, Any

# --- Project Imports ---
from ..core.core_services import CoreAgentServices
# No direct need to import Goal/Pipeline handlers, access via services

class EventHandler:
    """
    Handles events from an async queue using CoreAgentServices.
    Can trigger agent execution, goal initiation, or pipeline runs based on event type.
    """
    def __init__(self, services: CoreAgentServices, event_queue: asyncio.Queue):
        self.services = services
        self.event_queue = event_queue
        self.logger = logging.getLogger(__name__)
        # Store the task reference for potential external status checks/cancellation
        self._task: asyncio.Task | None = None
        self.logger.info("EventHandler initialized.")

    async def process_events(self):
        """Continuously processes events from the queue."""
        self.logger.info("Starting event processor loop...")
        while True:
            try:
                event_type, event_data = await self.event_queue.get()
                self.logger.info(f"Processing event: Type='{event_type}', Data='{event_data}'")

                task_prompt = None # For agent executor tasks
                goal_to_start = None # Tuple: (goal_type, goal_description)
                pipeline_to_start = None # Tuple: (pipeline_id, initial_context)

                task_id_base = f"event_{event_type}_{int(time.time())}"

                # --- Logic to convert event to task/goal/pipeline ---

                if event_type == "new_file":
                    file_path = event_data.get("file_path")
                    if file_path:
                        rel_path = os.path.relpath(file_path, self.services.workspace_dir) if file_path.startswith(self.services.workspace_dir) else file_path
                        filename = os.path.basename(rel_path)
                        task_id_base = f"event_newfile_{filename.split('.')[0]}"

                        # Example: Trigger knowledge addition directly via tool prompt
                        if filename.endswith(('.txt', '.pdf', '.md', '.docx')):
                             task_prompt = f"A new document '{filename}' appeared at '{rel_path}'. Add its content to the knowledge base using the 'add_knowledge_base_document' tool."
                        # Example: Trigger a pipeline for CSV analysis
                        elif filename.endswith('.csv') and "csv_analysis_report" in self.services.pipeline_handler._pipelines: # Check if pipeline exists
                             pipeline_to_start = ("csv_analysis_report", {"file_path": rel_path}) # Pass relative path
                        else:
                             task_prompt = f"A new file '{filename}' appeared at '{rel_path}'. Determine its type and summarize its content if possible."
                    else: self.logger.warning("New file event missing 'file_path'.")

                elif event_type == "api_trigger":
                    payload = event_data.get("payload", {})
                    endpoint = event_data.get("endpoint", "unknown")
                    action = payload.get("action", "process")
                    target = payload.get("target", "data")
                    task_id_base = f"event_api_{endpoint}_{action}"

                    # Example: Trigger a goal based on API payload
                    if action == "start_monitor" and "url" in payload:
                         goal_to_start = ("website_monitor", f"Monitor website {payload['url']} continuously.")
                    # Example: Trigger a pipeline
                    elif action == "run_pipeline" and "pipeline_id" in payload:
                         pipeline_to_start = (payload["pipeline_id"], payload.get("context", {}))
                    # Default: Use agent executor to interpret
                    else:
                         task_prompt = f"Received API trigger on endpoint '{endpoint}' requesting action '{action}' on '{target}' with details: {json.dumps(payload)}. Execute this request."

                elif event_type == "monitor_alert":
                    # Handle alerts from MonitoringHandler
                    level = event_data.get("level", "info")
                    message = event_data.get("message", "Monitor alert received.")
                    monitor_id = event_data.get("monitor_id", "unknown")
                    task_id_base = f"event_monitor_{monitor_id}"
                    # Example: Log critical alerts to memory, maybe use agent for complex alerts
                    if level == "critical":
                         task_prompt = f"CRITICAL ALERT from monitor '{monitor_id}': {message}. Analyze the situation based on the check result: {json.dumps(event_data.get('check_result'))} and suggest immediate actions."
                         await self.services.add_memory(f"CRITICAL ALERT: {message}", "system_monitor", task_id_base)
                    else:
                         # Just log non-critical alerts to memory
                         await self.services.add_memory(f"Monitor Alert ({level}): {message}", "system_monitor", task_id_base)
                         self.logger.info(f"Logged monitor alert ({level}) for '{monitor_id}'.")
                         # No further agent action needed for non-critical in this example

                # Add more event types here...
                # Example: Event specifically designed to start a goal
                elif event_type == "start_goal_request":
                     goal_type = event_data.get("goal_type")
                     goal_desc = event_data.get("goal_description")
                     if goal_type and goal_desc:
                          goal_to_start = (goal_type, goal_desc)
                     else:
                          self.logger.warning("start_goal_request event missing goal_type or goal_description.")

                # Example: Event specifically designed to start a pipeline
                elif event_type == "start_pipeline_request":
                     pipeline_id = event_data.get("pipeline_id")
                     initial_context = event_data.get("initial_context", {})
                     if pipeline_id:
                          pipeline_to_start = (pipeline_id, initial_context)
                     else:
                          self.logger.warning("start_pipeline_request event missing pipeline_id.")


                # --- Execute Action ---

                if goal_to_start:
                    # Start a Goal
                    if not self.services.goal_handler:
                         self.logger.error(f"Cannot start goal for event {event_type}: Goal Handler not available.")
                    else:
                         goal_type, goal_desc = goal_to_start
                         self.logger.info(f"Event '{event_type}' triggering start of goal '{goal_type}'...")
                         result = await self.services.goal_handler.start_goal(goal_desc, goal_type)
                         if result.get("goal_id"):
                              self.logger.info(f"Goal '{result['goal_id']}' started successfully.")
                              await self.services.add_memory(f"Event triggered start of goal {result['goal_id']} ({goal_type}).", "system_event", task_id_base)
                         else:
                              error_msg = result.get("error", "Unknown error starting goal.")
                              self.logger.error(f"Failed to start goal for event {event_type}: {error_msg}")
                              await self.services.add_memory(f"Event failed to trigger goal {goal_type}. Error: {error_msg}", "system_event_error", task_id_base)

                elif pipeline_to_start:
                    # Start a Pipeline
                    if not self.services.pipeline_handler:
                         self.logger.error(f"Cannot start pipeline for event {event_type}: Pipeline Handler not available.")
                    else:
                         pipeline_id, initial_context = pipeline_to_start
                         self.logger.info(f"Event '{event_type}' triggering start of pipeline '{pipeline_id}'...")
                         result = await self.services.pipeline_handler.start_pipeline(pipeline_id, initial_context)
                         if result.get("run_id"):
                              self.logger.info(f"Pipeline run '{result['run_id']}' started successfully.")
                              await self.services.add_memory(f"Event triggered start of pipeline run {result['run_id']} ({pipeline_id}).", "system_event", task_id_base)
                         else:
                              error_msg = result.get("error", "Unknown error starting pipeline.")
                              self.logger.error(f"Failed to start pipeline for event {event_type}: {error_msg}")
                              await self.services.add_memory(f"Event failed to trigger pipeline {pipeline_id}. Error: {error_msg}", "system_event_error", task_id_base)

                elif task_prompt:
                    # Execute using shared agent executor
                    if not self.services.agent_executor:
                         self.logger.error(f"Cannot process event task {task_id_base}: Agent Executor not available.")
                    else:
                         self.logger.info(f"Executing event task via Agent Executor ({task_id_base}): {task_prompt}")
                         try:
                             response = await self.services.agent_executor.ainvoke({"input": task_prompt})
                             output = response.get("output", "Event task produced no output.")
                             self.logger.info(f"Event task ({task_id_base}) completed. Output: {output[:200]}...")
                             await self.services.add_memory(f"Event Task: {task_prompt}\nResult: {output}", "system_event", task_id_base)
                         except Exception as task_err:
                              self.logger.error(f"Error executing event task {task_id_base}: {task_err}", exc_info=self.services.verbose)
                              await self.services.add_memory(f"Event Task Failed: {task_prompt}\nError: {task_err}", "system_event_error", task_id_base)
                else:
                    self.logger.warning(f"No action (agent task, goal, or pipeline) determined for event type '{event_type}'. Skipping.")

                self.event_queue.task_done() # Mark task as completed

            except asyncio.CancelledError:
                 self.logger.info("Event processor loop cancelled.")
                 break
            except Exception as e:
                 self.logger.error(f"Critical error in event processor loop: {e}", exc_info=True)
                 # Avoid crashing the loop, maybe add a delay before retrying?
                 # Ensure task_done is called even on error if appropriate for your queue logic
                 try: self.event_queue.task_done()
                 except ValueError: pass # If task_done already called or queue state invalid
                 await asyncio.sleep(5)

    def start(self):
        """Starts the event processing loop as a background task."""
        if self._task and not self._task.done():
            self.logger.warning("Event processor task already running.")
            return
        self.logger.info("Creating and starting event processor task...")
        self._task = asyncio.create_task(self.process_events())

    async def stop(self):
        """Stops the event processing loop."""
        self.logger.info("Stopping event processor loop...")
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                self.logger.info("Event processor task cancelled successfully.")
            except Exception as e:
                 self.logger.error(f"Error encountered while stopping event processor: {e}", exc_info=True)
        else:
            self.logger.info("Event processor task was not running or already finished.")
        self._task = None

