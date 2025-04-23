# handlers/monitoring_handler.py

import logging
import asyncio
import time
import os # Added os import
import yaml # Added yaml import
import re # Added for SQL result parsing example
import json # Added for tool call fallback example
import httpx # Added for website check example (if used)
from typing import Dict, Any, Optional, List, Callable

# --- Project Imports ---
from ..core.core_services import CoreAgentServices
# Import config to get CONFIG_DIR
from ..utils import config

class MonitoringHandler:
    """
    Proactively monitors specific conditions defined in config/monitors.yaml
    and triggers actions based on rules. Runs its own check loop independently.
    """
    def __init__(self,
                 services: CoreAgentServices,
                 event_queue: asyncio.Queue,
                 check_interval_seconds: int = 60):
        """
        Initializes the MonitoringHandler.

        Args:
            services: The shared CoreAgentServices instance.
            event_queue: The asyncio queue to enqueue events for the EventHandler.
            check_interval_seconds: How often to run all monitoring checks (in seconds).
        """
        self.services = services
        self.event_queue = event_queue
        self.check_interval_seconds = check_interval_seconds
        self.logger = logging.getLogger(__name__)
        self._monitor_loop_task: Optional[asyncio.Task] = None

        # Load definitions during initialization
        self._monitors: List[Dict[str, Any]] = self._load_monitors()

        # --- State Tracking for Monitors ---
        self._monitor_states: Dict[str, str] = {}
        self._monitor_failure_counts: Dict[str, int] = {}
        self._max_check_failures = 3 # Max consecutive failures before logging error
        # Initialize states for loaded monitors
        for monitor in self._monitors:
            # Only track state for enabled monitors with an ID
            if monitor.get("enabled", False):
                monitor_id = monitor.get("id")
                if monitor_id:
                    self._monitor_states[monitor_id] = "UNKNOWN" # Initial state
                    self._monitor_failure_counts[monitor_id] = 0
                else:
                     # This warning should ideally be caught during loading, but double-check
                     self.logger.warning(f"Monitor definition missing 'id': {monitor}. State tracking disabled.")

        self.logger.info(f"MonitoringHandler initialized with check interval: {check_interval_seconds}s")

    def _load_monitors(self) -> List[Dict[str, Any]]:
        """
        Loads monitor definitions from the monitors.yaml file.
        """
        self.logger.info("Loading monitor definitions from YAML file...")
        definitions = {}
        # Construct the path to the YAML file using CONFIG_DIR from utils.config
        monitors_file_path = os.path.join(config.CONFIG_DIR, "monitors.yaml")

        try:
            if not os.path.exists(monitors_file_path):
                 self.logger.warning(f"Monitors configuration file not found at: {monitors_file_path}. No monitors loaded.")
                 return []

            with open(monitors_file_path, 'r', encoding='utf-8') as f:
                definitions = yaml.safe_load(f)
                if not isinstance(definitions, dict):
                     self.logger.error(f"Failed to load monitors: YAML content in '{monitors_file_path}' is not a valid dictionary (mapping).")
                     return []

            # Convert dictionary to list and validate basic structure
            monitor_list = []
            for monitor_id_key, monitor_config in definitions.items():
                if not isinstance(monitor_config, dict):
                    self.logger.warning(f"Monitor definition for key '{monitor_id_key}' is not a dictionary. Skipping.")
                    continue
                # Ensure ID is present, using the key as fallback/default
                if "id" not in monitor_config:
                    monitor_config["id"] = monitor_id_key
                elif monitor_config["id"] != monitor_id_key:
                     # Allow ID in config to override the key if needed, but log it
                     self.logger.info(f"Monitor key '{monitor_id_key}' differs from config ID '{monitor_config['id']}'. Using config ID.")

                monitor_id = monitor_config["id"] # Use the final ID

                # Basic validation: check for required top-level keys
                if not all(k in monitor_config for k in ["check", "rule", "action"]):
                    self.logger.warning(f"Monitor '{monitor_id}' is missing required keys (check, rule, action). Skipping.")
                    continue

                monitor_list.append(monitor_config)

            loaded_count = len(monitor_list)
            self.logger.info(f"Successfully loaded and validated {loaded_count} monitor definitions from {monitors_file_path}.")
            return monitor_list

        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing monitors YAML file '{monitors_file_path}': {e}", exc_info=True)
            return []
        except FileNotFoundError:
             # This case is handled by the os.path.exists check above, but keep for safety
             self.logger.error(f"Monitors configuration file not found at: {monitors_file_path}")
             return []
        except Exception as e:
            self.logger.error(f"An unexpected error occurred loading monitors from '{monitors_file_path}': {e}", exc_info=True)
            return []

    async def _perform_check(self, monitor_config: Dict[str, Any]) -> Dict[str, Any]:
        """Performs a single check based on its configuration."""
        check_config = monitor_config.get("check", {})
        check_type = check_config.get("type")
        monitor_id = monitor_config.get("id", "unknown_monitor")
        self.logger.debug(f"Performing check for monitor '{monitor_id}' (type: {check_type})")
        result = {"status": "error", "value": None, "error_message": "Check type not implemented or invalid config"} # Default error result

        try:
            if check_type == "sql":
                query = check_config.get("query")
                if not query: raise ValueError("SQL check is missing 'query'.")
                if not self.services.sql_agent: raise RuntimeError("SQL Agent not available.")

                sql_response = await self.services._run_sync_in_thread(self.services.sql_agent.run, query)

                if sql_response.get("error"):
                    raise RuntimeError(f"SQL query failed: {sql_response['error']}")

                raw_result = sql_response.get("result")
                self.logger.debug(f"SQL check raw result: {raw_result}")
                try:
                    # Attempt to parse numeric value
                    match = re.search(r'\d+(\.\d+)?', str(raw_result))
                    parsed_value = float(match.group(0)) if match else None
                    # Use the field name specified in the rule config as the key in the result dict
                    rule_field = monitor_config.get("rule", {}).get("field", "value")
                    result = {"status": "ok", "value": parsed_value, rule_field: parsed_value, "raw_result": raw_result}
                except Exception as parse_err:
                    self.logger.warning(f"Could not parse numeric value from SQL result for '{monitor_id}': {parse_err}. Using raw result.")
                    rule_field = monitor_config.get("rule", {}).get("field", "value")
                    result = {"status": "ok", "value": raw_result, rule_field: raw_result, "raw_result": raw_result}

            elif check_type == "db_ping":
                if not self.services.sql_agent or not self.services.sql_agent.db:
                     raise RuntimeError("SQL Agent or DB connection not available for ping.")
                try:
                    engine = self.services.sql_agent.db.engine
                    await self.services._run_sync_in_thread(engine.connect) # Test connection
                    result = {"status": "ok", "value": "Connected"}
                except Exception as db_err:
                     raise RuntimeError(f"DB Ping failed: {db_err}") from db_err

            elif check_type == "tool_call":
                tool_name = check_config.get("tool_name")
                params = check_config.get("params", {})
                if not tool_name: raise ValueError("Tool call check is missing 'tool_name'.")

                # Find async wrapper first
                tool_func: Optional[Callable] = getattr(self.services, f"_{tool_name}_async", None)
                if not tool_func or not asyncio.iscoroutinefunction(tool_func):
                     # Fallback: Use agent executor
                     self.logger.warning(f"Direct async tool wrapper '_{tool_name}_async' not found for check. Using agent executor.")
                     if not self.services.agent_executor: raise RuntimeError("Agent Executor not available.")
                     prompt = f"Run the tool '{tool_name}' with parameters: {json.dumps(params)}"
                     tool_response = await self.services.agent_executor.ainvoke({"input": prompt})
                     raw_tool_result = tool_response.get("output", "Tool execution via agent failed or produced no output.")
                     # Attempt basic JSON parsing
                     try: parsed_tool_result = json.loads(raw_tool_result)
                     except: parsed_tool_result = raw_tool_result
                     result = {"status": "ok", "value": parsed_tool_result, "raw_result": raw_tool_result}
                else:
                     # Call direct async wrapper
                     self.logger.debug(f"Calling direct tool wrapper '_{tool_name}_async' with params: {params}")
                     tool_result = await tool_func(**params)
                     # Structure the result
                     if isinstance(tool_result, dict):
                          result = {"status": "ok", **tool_result}
                          if "value" not in result: result["value"] = tool_result # Use whole dict if no 'value'
                     else:
                          result = {"status": "ok", "value": tool_result}

            # Add other check types here (e.g., file existence, API call)

            # Check completed successfully
            if monitor_id in self._monitor_failure_counts: self._monitor_failure_counts[monitor_id] = 0
            return result

        except Exception as e:
            self.logger.error(f"Check failed for monitor '{monitor_id}': {e}", exc_info=self.services.verbose)
            # Increment failure count if tracking this monitor ID
            if monitor_id in self._monitor_failure_counts:
                 self._monitor_failure_counts[monitor_id] += 1
                 if self._monitor_failure_counts[monitor_id] >= self._max_check_failures:
                      self.logger.error(f"Monitor '{monitor_id}' check has failed {self._monitor_failure_counts[monitor_id]} consecutive times.")
                      # Optionally trigger a specific 'check_failed' alert here
            result["status"] = "error"
            result["error_message"] = str(e)
            return result


    def _evaluate_rule(self, check_result: Dict[str, Any], rule_config: Dict[str, Any]) -> bool:
        """Evaluates if the check result meets the rule's condition."""
        rule_type = rule_config.get("type")
        # Default field to 'value' if not specified in rule config
        field = rule_config.get("field", "value")
        actual_value = check_result.get(field)

        self.logger.debug(f"Evaluating rule type '{rule_type}' on field '{field}' with value '{actual_value}'")

        try:
            if rule_type == "threshold":
                operator = rule_config.get("operator")
                threshold_value = rule_config.get("value")
                if operator is None or threshold_value is None: raise ValueError("Threshold rule missing 'operator' or 'value'.")
                try: num_actual = float(actual_value)
                except (ValueError, TypeError, AttributeError): raise ValueError(f"Cannot compare non-numeric value '{actual_value}' for threshold rule.")
                try: num_threshold = float(threshold_value)
                except (ValueError, TypeError): raise ValueError(f"Invalid threshold value '{threshold_value}'.")

                op_map = {">": lambda a, b: a > b, "<": lambda a, b: a < b,
                          ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
                          "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
                if operator not in op_map: raise ValueError(f"Unsupported operator '{operator}' in threshold rule.")
                return op_map[operator](num_actual, num_threshold)

            elif rule_type == "equality":
                expected_value = rule_config.get("value")
                # Compare as strings for flexibility, unless specific type needed
                return str(actual_value) == str(expected_value)

            elif rule_type == "contains":
                 substring = rule_config.get("value")
                 if substring is None: raise ValueError("Contains rule missing 'value'.")
                 return substring in str(actual_value)

            # Add other rule types (regex, not_contains, etc.)

            else:
                raise ValueError(f"Unsupported rule type: {rule_type}")

        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule_config}: {e}", exc_info=self.services.verbose)
            return False # Default to false if rule evaluation fails


    async def _trigger_action(self, monitor_config: Dict[str, Any], check_result: Dict[str, Any], is_alerting: bool):
        """Triggers the configured action (enqueue event or direct call)."""
        action_config = monitor_config.get("action", {})
        action_type = action_config.get("type")
        monitor_id = monitor_config.get("id", "unknown")
        description = monitor_config.get("description", "")
        current_status = "ALERT" if is_alerting else "OK"

        self.logger.info(f"Triggering action for monitor '{monitor_id}'. Condition met: {is_alerting}. Current State: {current_status}")

        try:
            if action_type == "enqueue_event":
                event_type = action_config.get("event_type", "monitor_event")
                event_data_template = action_config.get("event_data", {})

                # Format message using template and results
                message_template = event_data_template.get("message_template", "Monitor '{monitor_id}' status: {current_status}. Result: {check_result}")
                try:
                    # Include relevant fields in the format context
                    format_context = {
                        "monitor_id": monitor_id,
                        "description": description,
                        "current_status": current_status,
                        "check_result": check_result.get("value", "N/A"), # Use the primary value
                        "result_value": check_result.get("value", "N/A"), # Alias for clarity
                        "raw_result": check_result.get("raw_result", check_result.get("value", "N/A")),
                        "error_message": check_result.get("error_message", ""),
                        # Add other fields from check_result if needed
                        **(check_result if isinstance(check_result, dict) else {})
                    }
                    message = message_template.format(**format_context)
                except KeyError as fmt_err:
                     self.logger.warning(f"KeyError formatting message template for '{monitor_id}': {fmt_err}. Using raw template.")
                     message = message_template # Fallback

                # Construct final event data
                event_data = {
                    **event_data_template, # Include base template fields (like level)
                    "monitor_id": monitor_id,
                    "monitor_description": description,
                    "status": current_status,
                    "timestamp": time.time(),
                    "check_result": check_result, # Include full check result
                    "message": message, # Overwrite template with formatted message
                    # "message_template": None # Optionally remove original template
                }

                self.logger.info(f"Enqueuing event type '{event_type}' for monitor '{monitor_id}'")
                await self.event_queue.put((event_type, event_data))

            elif action_type == "direct_call":
                tool_name = action_config.get("tool_name")
                params_template = action_config.get("params", {})
                if not tool_name: raise ValueError("Direct call action missing 'tool_name'.")

                # Format parameters using template and results
                params = {}
                format_context = { # Same context as for enqueue_event message
                    "monitor_id": monitor_id, "description": description, "current_status": current_status,
                    "check_result": check_result.get("value", "N/A"), "result_value": check_result.get("value", "N/A"),
                    "raw_result": check_result.get("raw_result", check_result.get("value", "N/A")),
                    "error_message": check_result.get("error_message", ""),
                    **(check_result if isinstance(check_result, dict) else {})
                }
                try:
                    for key, value_template in params_template.items():
                        if isinstance(value_template, str):
                            params[key] = value_template.format(**format_context)
                        else:
                            params[key] = value_template # Pass non-strings directly
                except KeyError as fmt_err:
                     self.logger.error(f"KeyError formatting parameters for direct call action '{tool_name}': {fmt_err}. Aborting action.")
                     return # Don't proceed if params can't be formatted

                # Find and call the tool (similar to _perform_check)
                tool_func: Optional[Callable] = getattr(self.services, f"_{tool_name}_async", None)
                if tool_func and asyncio.iscoroutinefunction(tool_func):
                    self.logger.info(f"Executing direct tool call action: '{tool_name}' with params: {params}")
                    await tool_func(**params)
                else:
                    # Fallback: Use agent executor (less ideal for direct actions)
                    self.logger.warning(f"Direct async tool wrapper '_{tool_name}_async' not found. Using agent executor for action.")
                    if not self.services.agent_executor: raise RuntimeError("Agent Executor not available.")
                    prompt = f"Run the tool '{tool_name}' with parameters: {json.dumps(params)}"
                    await self.services.agent_executor.ainvoke({"input": prompt})
                    # Note: We don't typically wait for or process the result of a direct action call here.

            else:
                self.logger.warning(f"Unsupported action type '{action_type}' for monitor '{monitor_id}'.")

        except Exception as e:
            self.logger.error(f"Error triggering action for monitor '{monitor_id}': {e}", exc_info=self.services.verbose)


    async def _monitor_loop(self):
        """The main asynchronous loop that runs checks periodically."""
        self.logger.info("Starting monitoring loop...")
        await asyncio.sleep(5) # Initial delay before first check

        while True:
            start_time = time.monotonic()
            self.logger.info("Running monitoring checks...")
            # Filter enabled monitors from the potentially reloaded list
            active_monitors = [m for m in self._monitors if m.get("enabled", False)]
            self.logger.debug(f"Found {len(active_monitors)} enabled monitors.")

            if not active_monitors:
                 self.logger.info("No enabled monitors found. Sleeping.")
                 # Prevent busy-waiting if no monitors are enabled
                 await asyncio.sleep(self.check_interval_seconds)
                 continue

            check_tasks = []
            monitor_map = {} # Map task back to monitor config
            for monitor_config in active_monitors:
                 # Ensure monitor has an ID before creating task
                 monitor_id = monitor_config.get("id")
                 if not monitor_id:
                      self.logger.warning(f"Skipping monitor check as it lacks an 'id': {monitor_config}")
                      continue
                 task = asyncio.create_task(self._perform_check(monitor_config))
                 check_tasks.append(task)
                 monitor_map[task] = monitor_config

            # Run checks concurrently
            results = await asyncio.gather(*check_tasks, return_exceptions=True)

            # Process results
            action_tasks = []
            for i, result_or_exc in enumerate(results):
                task = check_tasks[i]
                monitor_config = monitor_map[task]
                monitor_id = monitor_config["id"] # Should exist if task was created
                check_result: Dict[str, Any]

                if isinstance(result_or_exc, Exception):
                    self.logger.error(f"Check task for monitor '{monitor_id}' raised an exception: {result_or_exc}", exc_info=self.services.verbose)
                    check_result = {"status": "error", "value": None, "error_message": str(result_or_exc)}
                else:
                    check_result = result_or_exc # Should be the dict from _perform_check

                # Evaluate rule only if check didn't error catastrophically
                # or if rule is configured to trigger on check error
                rule_met = False
                rule_config = monitor_config.get("rule", {})
                trigger_on_error = rule_config.get("trigger_on_check_error", False)

                if check_result.get("status") != "error" or trigger_on_error:
                     if rule_config:
                         rule_met = self._evaluate_rule(check_result, rule_config)
                         self.logger.debug(f"Monitor '{monitor_id}' rule evaluation result: {rule_met}")
                     else:
                          self.logger.debug(f"Monitor '{monitor_id}' has no rule defined.")
                else:
                     self.logger.warning(f"Skipping rule evaluation for '{monitor_id}' due to check error (and trigger_on_check_error is false).")


                # --- State Change Detection & Action Triggering ---
                previous_state = self._monitor_states.get(monitor_id, "UNKNOWN")
                current_state = "ALERT" if rule_met else "OK"

                if current_state != previous_state:
                    self.logger.info(f"State change detected for monitor '{monitor_id}': {previous_state} -> {current_state}")
                    self._monitor_states[monitor_id] = current_state
                    # Trigger action on state change
                    # Check if action should trigger on alert or resolve
                    action_config = monitor_config.get("action", {})
                    alert_on_resolve = action_config.get("alert_on_resolve", False)
                    is_alerting = (current_state == "ALERT")
                    is_resolving = (current_state == "OK")

                    should_trigger = is_alerting or (is_resolving and alert_on_resolve)

                    if should_trigger:
                         action_tasks.append(
                             asyncio.create_task(self._trigger_action(monitor_config, check_result, is_alerting))
                         )
                    else:
                         self.logger.debug(f"No action triggered for '{monitor_id}' state change to {current_state} based on config.")
                else:
                    self.logger.debug(f"No state change for monitor '{monitor_id}'. Current state: {current_state}")

            # Await any triggered actions if needed (or let them run in background)
            if action_tasks:
                # Wait for actions to complete, log any errors from them
                action_results = await asyncio.gather(*action_tasks, return_exceptions=True)
                for action_result in action_results:
                    if isinstance(action_result, Exception):
                         self.logger.error(f"Error occurred during triggered monitor action: {action_result}", exc_info=action_result)


            # --- Loop Timing ---
            elapsed = time.monotonic() - start_time
            sleep_duration = max(0, self.check_interval_seconds - elapsed)
            self.logger.info(f"Monitoring checks completed in {elapsed:.2f}s. Sleeping for {sleep_duration:.2f}s.")
            try:
                await asyncio.sleep(sleep_duration)
            except asyncio.CancelledError:
                self.logger.info("Monitoring loop cancelled during sleep.")
                break # Exit loop if cancelled

    def start(self):
        """Loads/reloads monitors and starts the monitoring loop."""
        if self._monitor_loop_task and not self._monitor_loop_task.done():
            self.logger.warning("Monitoring loop already running.")
            return

        # Reload definitions on start in case config changed (if not using live reload)
        self._monitors = self._load_monitors()
        # Re-initialize states based on currently loaded monitors
        self._monitor_states = {}
        self._monitor_failure_counts = {}
        for monitor in self._monitors:
            if monitor.get("enabled", False) and (monitor_id := monitor.get("id")):
                self._monitor_states[monitor_id] = "UNKNOWN"
                self._monitor_failure_counts[monitor_id] = 0

        # Check if any monitors are actually enabled before starting loop
        if not any(m.get("enabled", False) for m in self._monitors):
             self.logger.warning("No monitors enabled after loading config. Monitoring loop will not start.")
             return

        self.logger.info("Creating and starting monitoring loop task...")
        self._monitor_loop_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stops the monitoring loop task."""
        self.logger.info("Stopping monitoring loop...")
        if self._monitor_loop_task and not self._monitor_loop_task.done():
            self._monitor_loop_task.cancel()
            try:
                await self._monitor_loop_task
            except asyncio.CancelledError:
                self.logger.info("Monitoring loop task cancelled successfully.")
            except Exception as e:
                 # Log error but continue shutdown
                 self.logger.error(f"Error encountered while stopping monitoring loop: {e}", exc_info=True)
        else:
            self.logger.info("Monitoring loop was not running or already finished.")
        self._monitor_loop_task = None # Clear task reference

