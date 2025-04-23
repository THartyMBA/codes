# handlers/admin_handler.py

import logging
import asyncio
import os
import importlib # For potential (careful) config reload attempts
from typing import Dict, Any, Optional, List

# --- Project Imports ---
from ..core.core_services import CoreAgentServices
from ..utils import config
# Import other handlers to potentially query their state (optional)
from .scheduler_handler import SchedulerHandler
from .event_handler import EventHandler
from .goal_handler import GoalHandler
from .pipeline_handler import PipelineHandler
from .monitoring_handler import MonitoringHandler # Added
from .knowledge_handler import KnowledgeHandler # Added


class AdminHandler:
    """
    Handles administrative tasks for the agent system.
    Provides functions for monitoring, configuration, and basic control.
    Access to these functions should be secured at the API layer.
    """
    def __init__(self,
                 services: CoreAgentServices,
                 # Pass references to handlers managed externally (like api.py)
                 scheduler_handler: Optional[SchedulerHandler] = None,
                 event_handler: Optional[EventHandler] = None,
                 # Handlers initialized within CoreServices are accessed via services
                 # goal_handler: Optional[GoalHandler] = None, # Access via services.goal_handler
                 # pipeline_handler: Optional[PipelineHandler] = None, # Access via services.pipeline_handler
                 # monitoring_handler: Optional[MonitoringHandler] = None # Access via services.monitoring_handler
                 # knowledge_handler: Optional[KnowledgeHandler] = None # Access via services.knowledge_handler
                 ):
        """
        Initializes the AdminHandler.

        Args:
            services: The shared CoreAgentServices instance.
            scheduler_handler: Reference to the SchedulerHandler instance (if managed externally).
            event_handler: Reference to the EventHandler instance (if managed externally).
        """
        self.services = services
        # Store references to externally managed handlers if provided
        self.ext_scheduler_handler = scheduler_handler
        self.ext_event_handler = event_handler
        # Access internally managed handlers via self.services
        self.logger = logging.getLogger(__name__)
        self.logger.info("AdminHandler initialized.")

    async def get_system_overview(self) -> Dict[str, Any]:
        """Provides a high-level overview of the system status."""
        self.logger.debug("Getting system overview...")
        overview = {
            "core_services_status": "Initialized" if self.services else "Not Initialized",
            "llm_model": config.OLLAMA_MODEL,
            "db_type": config.DB_TYPE,
            "vector_db_path": config.CHROMA_DB_PATH,
            "workspace_path": config.WORKSPACE_DIR,
            "handlers_status": {}
        }
        # Get status from handlers
        # Use externally provided refs OR refs from CoreServices
        scheduler_ref = self.ext_scheduler_handler # Use external ref if provided
        if scheduler_ref and scheduler_ref.scheduler:
            overview["handlers_status"]["scheduler"] = {
                "running": scheduler_ref.scheduler.running,
                "job_count": len(scheduler_ref.scheduler.get_jobs())
            }
        event_ref = self.ext_event_handler # Use external ref if provided
        if event_ref and event_ref.event_queue:
             overview["handlers_status"]["event_processor"] = {
                 # Check if the task exists and isn't done (best guess for running)
                 "running": hasattr(event_ref, '_task') and event_ref._task and not event_ref._task.done(),
                 "queue_size": event_ref.event_queue.qsize()
             }
        # Access handlers initialized within CoreServices
        if self.services.goal_handler:
             overview["handlers_status"]["goal_handler"] = {
                 "active_goals": len(self.services.goal_handler.running_goal_tasks)
             }
        if self.services.pipeline_handler:
             overview["handlers_status"]["pipeline_handler"] = {
                 "active_pipelines": len(self.services.pipeline_handler.running_pipelines),
                 "completed_pipelines": len(self.services.pipeline_handler.pipeline_results) # Includes failed
             }
        if self.services.monitoring_handler:
             overview["handlers_status"]["monitoring_handler"] = {
                 "running": self.services.monitoring_handler._monitor_loop_task is not None and not self.services.monitoring_handler._monitor_loop_task.done(),
                 "monitor_count": len(self.services.monitoring_handler._monitors)
             }
        if self.services.knowledge_handler:
             # Knowledge handler doesn't have a 'running' state, maybe check collection status?
             try:
                  count = self.services.collection.count()
                  overview["handlers_status"]["knowledge_handler"] = {"status": "OK", "document_chunks": count}
             except Exception as e:
                  overview["handlers_status"]["knowledge_handler"] = {"status": "Error", "detail": str(e)}


        return overview

    async def trigger_knowledge_reindex(self, source_path: str, force_add: bool = False) -> Dict[str, Any]:
        """
        Triggers the knowledge ingestion process using the KnowledgeHandler.

        Args:
            source_path: Path relative to workspace or absolute path to index.
            force_add: If True, attempts to add even if memory suggests it was added before.

        Returns:
            Status dictionary from the KnowledgeHandler.
        """
        self.logger.info(f"Admin request: Triggering knowledge re-index for source: {source_path} (Force: {force_add})")

        if not self.services.knowledge_handler:
            return {"status": "error", "message": "Knowledge Handler not available."}

        # Optional: Check memory before forcing (moved check logic here for clarity)
        if not force_add:
            try:
                mem_query = f"Ingested.*{os.path.basename(source_path)}"
                mem_search = await self.services.retrieve_memory(query=mem_query, user_id="system_knowledge", limit=1) # Check system memory
                if mem_search:
                    msg = f"Knowledge from '{source_path}' likely already ingested (found in memory). Use force_add=True to override."
                    self.logger.warning(msg)
                    return {"status": "skipped", "message": msg}
            except Exception as e:
                 self.logger.warning(f"Could not check memory before re-indexing: {e}")

        # Trigger the KnowledgeHandler's add_source method
        try:
            result_dict = await self.services.knowledge_handler.add_source(source_path)
            self.logger.info(f"Knowledge indexing result for '{source_path}': {result_dict}")
            # Return the dictionary directly from the handler
            return result_dict
        except Exception as e:
            self.logger.error(f"Error during knowledge re-indexing trigger for '{source_path}': {e}", exc_info=self.services.verbose)
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}

    async def clear_user_memory(self, user_id: str) -> Dict[str, Any]:
        """
        Clears the conversation memory for a specific user_id in Mem0.
        WARNING: This is destructive. Requires Mem0 client to support deletion.
        """
        # (Implementation remains the same - relies on potential future Mem0 feature)
        self.logger.warning(f"Admin request: Attempting to clear memory for user_id: {user_id}")
        try:
            # --- Replace with actual Mem0 deletion call if/when available ---
            msg = f"Memory deletion for user_id '{user_id}' is not currently supported by the Mem0 client or this handler."
            self.logger.error(msg)
            return {"status": "error", "message": msg}
            # --- End Placeholder ---
        except Exception as e:
            self.logger.error(f"Error attempting to clear memory for user_id '{user_id}': {e}", exc_info=self.services.verbose)
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}

    async def get_log_file_paths(self) -> Dict[str, Any]:
        """Returns the configured paths for log files."""
        # (Implementation remains the same)
        self.logger.debug("Admin request: Getting log file paths.")
        paths = {}
        try:
            log_dir = os.getenv("LOG_DIRECTORY", os.path.join(config.WORKSPACE_DIR, "logs"))
            log_filename = os.getenv("LOG_FILENAME", "gen_ai_app.log")
            paths["main_log_file"] = os.path.join(log_dir, log_filename)
        except Exception as e:
            self.logger.warning(f"Could not determine main log file path: {e}")
            paths["main_log_file"] = "Error determining path"
        paths["autonomous_task_logs"] = os.path.join(config.WORKSPACE_DIR, "autonomous_logs") # Keep this if used by handlers
        return {"status": "success", "log_paths": paths}

    async def check_system_health(self) -> Dict[str, Any]:
        """Performs basic health checks on core components."""
        # (Implementation needs update for knowledge handler check)
        self.logger.info("Admin request: Performing system health checks...")
        health_status = {
            "overall_status": "OK",
            "checks": {}
        }
        all_ok = True

        # 1. LLM Check
        try:
            await self.services.llm.ainvoke("Respond with OK.")
            health_status["checks"]["llm_service"] = {"status": "OK"}
        except Exception as e:
            self.logger.error(f"Health Check Failed: LLM service error: {e}", exc_info=self.services.verbose)
            health_status["checks"]["llm_service"] = {"status": "ERROR", "detail": str(e)}
            all_ok = False

        # 2. Database Check
        if self.services.sql_agent and self.services.sql_agent.db:
            try:
                engine = self.services.sql_agent.db.engine
                await self.services._run_sync_in_thread(engine.connect)
                health_status["checks"]["database_connection"] = {"status": "OK"}
            except Exception as e:
                self.logger.error(f"Health Check Failed: Database connection error: {e}", exc_info=self.services.verbose)
                health_status["checks"]["database_connection"] = {"status": "ERROR", "detail": str(e)}
                all_ok = False
        else:
            health_status["checks"]["database_connection"] = {"status": "NOT_CONFIGURED"}

        # 3. Vector DB Check (Use KnowledgeHandler)
        if self.services.knowledge_handler:
            try:
                # Perform a simple retrieve operation
                await self.services.knowledge_handler.retrieve("health check query", n_results=1)
                health_status["checks"]["vector_db"] = {"status": "OK"}
            except Exception as e:
                self.logger.error(f"Health Check Failed: Vector DB error: {e}", exc_info=self.services.verbose)
                health_status["checks"]["vector_db"] = {"status": "ERROR", "detail": str(e)}
                all_ok = False
        else:
             health_status["checks"]["vector_db"] = {"status": "NOT_INITIALIZED"}
             all_ok = False # If handler missing, system isn't fully healthy

        # 4. Memory Check
        try:
            await self.services.retrieve_memory("health check query", user_id="system_health", limit=1)
            health_status["checks"]["memory_service"] = {"status": "OK"}
        except Exception as e:
            self.logger.error(f"Health Check Failed: Memory service error: {e}", exc_info=self.services.verbose)
            health_status["checks"]["memory_service"] = {"status": "ERROR", "detail": str(e)}
            all_ok = False

        # Update overall status
        if not all_ok:
            health_status["overall_status"] = "ERROR"

        self.logger.info(f"Health check completed. Overall status: {health_status['overall_status']}")
        return health_status

    async def reload_configuration(self) -> Dict[str, Any]:
        """
        Attempts to reload configuration. (EXPERIMENTAL/UNSAFE)
        """
        # (Implementation remains the same - still experimental)
        self.logger.warning("Admin request: Attempting configuration reload (EXPERIMENTAL/UNSAFE)...")
        try:
            importlib.reload(config)
            msg = "Configuration module reloaded. Restart recommended for changes to take full effect."
            self.logger.warning(msg)
            return {"status": "partial_success", "message": msg}
        except Exception as e:
            self.logger.error(f"Error attempting to reload configuration: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to reload configuration: {e}"}

    async def stop(self):
        """Placeholder stop method if needed for admin handler cleanup."""
        self.logger.info("AdminHandler stopping (no specific cleanup actions).")

