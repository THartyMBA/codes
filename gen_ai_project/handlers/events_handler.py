# handlers/event_handler.py

import asyncio
import logging
import json
import time
import os

# --- Project Imports ---
from ..core.core_services import CoreAgentServices

class EventHandler:
    """Handles events from an async queue using CoreAgentServices."""
    def __init__(self, services: CoreAgentServices, event_queue: asyncio.Queue):
        self.services = services
        self.event_queue = event_queue
        self.logger = logging.getLogger(__name__)
        self.logger.info("EventHandler initialized.")

    async def process_events(self):
        """Continuously processes events from the queue."""
        self.logger.info("Starting event processor loop...")
        while True:
            try:
                event_type, event_data = await self.event_queue.get()
                self.logger.info(f"Processing event: Type='{event_type}', Data='{event_data}'")

                task_prompt = None
                task_id_base = f"event_{event_type}_{int(time.time())}"

                # --- Logic to convert event to task prompt ---
                if event_type == "new_file":
                    file_path = event_data.get("file_path")
                    if file_path:
                        rel_path = os.path.relpath(file_path, self.services.workspace_dir) if file_path.startswith(self.services.workspace_dir) else file_path
                        filename = os.path.basename(rel_path)
                        # Example: Ask to add to knowledge base or analyze
                        if filename.endswith(('.txt', '.pdf', '.md', '.docx')):
                             task_prompt = f"A new document '{filename}' appeared at '{rel_path}'. Add its content to the knowledge base."
                        elif filename.endswith('.csv'):
                             task_prompt = f"A new CSV file '{filename}' appeared at '{rel_path}'. Load it, analyze its columns, and generate a summary report named '{filename}_summary.pdf'."
                        else:
                             task_prompt = f"A new file '{filename}' appeared at '{rel_path}'. Determine its type and take appropriate action (e.g., summarize if text)."
                        task_id_base = f"event_newfile_{filename.split('.')[0]}"
                    else: self.logger.warning("New file event missing 'file_path'.")

                elif event_type == "api_trigger":
                    payload = event_data.get("payload", {})
                    endpoint = event_data.get("endpoint", "unknown")
                    # Use LLM to interpret the payload and decide action? Or hardcode logic?
                    # Example: Simple interpretation
                    action = payload.get("action", "process")
                    target = payload.get("target", "data")
                    task_prompt = f"Received API trigger on endpoint '{endpoint}' requesting action '{action}' on '{target}' with details: {json.dumps(payload)}. Execute this request."
                    task_id_base = f"event_api_{endpoint}_{action}"

                # Add more event types here...

                # --- Execute task using shared agent executor ---
                if task_prompt:
                    if not self.services.agent_executor:
                         self.logger.error(f"Cannot process event {task_id_base}: Agent Executor not available.")
                         continue # Skip to next event

                    self.logger.info(f"Executing event task ({task_id_base}): {task_prompt}")
                    try:
                        response = await self.services.agent_executor.ainvoke({"input": task_prompt})
                        output = response.get("output", "Event task produced no output.")
                        self.logger.info(f"Event task ({task_id_base}) completed. Output: {output[:200]}...")
                        # Log result / add to system memory via self.services
                        await self.services.add_memory(f"Event Task: {task_prompt}\nResult: {output}", "system_event", task_id_base)
                    except Exception as task_err:
                         self.logger.error(f"Error executing event task {task_id_base}: {task_err}", exc_info=self.services.verbose)
                         await self.services.add_memory(f"Event Task Failed: {task_prompt}\nError: {task_err}", "system_event_error", task_id_base)

                else:
                    self.logger.warning(f"No task prompt generated for event type '{event_type}'. Skipping.")

                self.event_queue.task_done() # Mark task as completed

            except asyncio.CancelledError:
                 self.logger.info("Event processor loop cancelled.")
                 break
            except Exception as e:
                 self.logger.error(f"Critical error in event processor loop: {e}", exc_info=True)
                 # Avoid crashing the loop, maybe add a delay before retrying?
                 await asyncio.sleep(5)

