# handlers/scheduler_handler.py

import logging
import asyncio
import json # For potentially parsing complex args
from typing import Dict, Any, Optional, List
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# --- Project Imports ---
from ..core.core_services import CoreAgentServices

class SchedulerHandler:
    """
    Manages scheduled tasks using APScheduler and CoreAgentServices.
    Can schedule agent prompts, goal initiations, or pipeline runs.
    """
    def __init__(self, services: CoreAgentServices):
        self.services = services
        self.scheduler = AsyncIOScheduler(timezone="UTC") # Use UTC or your desired timezone
        self.logger = logging.getLogger(__name__)
        self.logger.info("SchedulerHandler initialized.")

    async def _execute_scheduled_action(self, action_type: str, task_id: str, **kwargs):
        """
        Generic function executed by the scheduler to perform different actions.

        Args:
            action_type: Type of action ('agent_prompt', 'start_goal', 'start_pipeline').
            task_id: The unique ID for this scheduled task instance.
            **kwargs: Arguments specific to the action type.
                      - For 'agent_prompt': expects 'prompt' (str)
                      - For 'start_goal': expects 'goal_type' (str), 'goal_description' (str)
                      - For 'start_pipeline': expects 'pipeline_id' (str), 'initial_context' (dict, optional)
        """
        self.logger.info(f"Executing scheduled action ({task_id}): Type='{action_type}', Args={kwargs}")

        if action_type == 'agent_prompt':
            prompt = kwargs.get('prompt')
            if not prompt:
                self.logger.error(f"Scheduled task {task_id} (agent_prompt) missing 'prompt' argument.")
                return
            if not self.services.agent_executor:
                self.logger.error(f"Cannot run scheduled task {task_id}: Agent Executor not available.")
                return
            try:
                self.logger.debug(f"Running agent prompt for task {task_id}...")
                response = await self.services.agent_executor.ainvoke({"input": prompt})
                output = response.get("output", "Scheduled agent prompt produced no output.")
                self.logger.info(f"Scheduled task ({task_id}) completed. Output: {output[:200]}...")
                await self.services.add_memory(f"Scheduled Task (Agent Prompt): {prompt}\nResult: {output}", "system_scheduler", task_id)
            except Exception as e:
                self.logger.error(f"Error executing scheduled agent prompt {task_id}: {e}", exc_info=self.services.verbose)
                await self.services.add_memory(f"Scheduled Task Failed (Agent Prompt): {prompt}\nError: {e}", "system_scheduler_error", task_id)

        elif action_type == 'start_goal':
            goal_type = kwargs.get('goal_type')
            goal_description = kwargs.get('goal_description')
            if not goal_type or not goal_description:
                self.logger.error(f"Scheduled task {task_id} (start_goal) missing 'goal_type' or 'goal_description'.")
                return
            if not self.services.goal_handler:
                self.logger.error(f"Cannot start scheduled goal {task_id}: Goal Handler not available.")
                return
            try:
                self.logger.debug(f"Starting goal '{goal_type}' for task {task_id}...")
                result = await self.services.goal_handler.start_goal(goal_description, goal_type)
                if result.get("goal_id"):
                    self.logger.info(f"Scheduled task ({task_id}) successfully started goal '{result['goal_id']}'.")
                    await self.services.add_memory(f"Scheduled Task triggered start of goal {result['goal_id']} ({goal_type}).", "system_scheduler", task_id)
                else:
                    error_msg = result.get("error", "Unknown error starting goal.")
                    self.logger.error(f"Scheduled task ({task_id}) failed to start goal '{goal_type}': {error_msg}")
                    await self.services.add_memory(f"Scheduled Task failed to trigger goal {goal_type}. Error: {error_msg}", "system_scheduler_error", task_id)
            except Exception as e:
                self.logger.error(f"Error executing scheduled goal start {task_id}: {e}", exc_info=self.services.verbose)
                await self.services.add_memory(f"Scheduled Task failed (Goal Start): {goal_description}\nError: {e}", "system_scheduler_error", task_id)

        elif action_type == 'start_pipeline':
            pipeline_id = kwargs.get('pipeline_id')
            initial_context = kwargs.get('initial_context', {}) # Default to empty dict
            if not pipeline_id:
                self.logger.error(f"Scheduled task {task_id} (start_pipeline) missing 'pipeline_id'.")
                return
            if not self.services.pipeline_handler:
                self.logger.error(f"Cannot start scheduled pipeline {task_id}: Pipeline Handler not available.")
                return
            try:
                self.logger.debug(f"Starting pipeline '{pipeline_id}' for task {task_id}...")
                result = await self.services.pipeline_handler.start_pipeline(pipeline_id, initial_context)
                if result.get("run_id"):
                    self.logger.info(f"Scheduled task ({task_id}) successfully started pipeline run '{result['run_id']}'.")
                    await self.services.add_memory(f"Scheduled Task triggered start of pipeline run {result['run_id']} ({pipeline_id}).", "system_scheduler", task_id)
                else:
                    error_msg = result.get("error", "Unknown error starting pipeline.")
                    self.logger.error(f"Scheduled task ({task_id}) failed to start pipeline '{pipeline_id}': {error_msg}")
                    await self.services.add_memory(f"Scheduled Task failed to trigger pipeline {pipeline_id}. Error: {error_msg}", "system_scheduler_error", task_id)
            except Exception as e:
                self.logger.error(f"Error executing scheduled pipeline start {task_id}: {e}", exc_info=self.services.verbose)
                await self.services.add_memory(f"Scheduled Task failed (Pipeline Start): {pipeline_id}\nError: {e}", "system_scheduler_error", task_id)

        else:
            self.logger.error(f"Scheduled task {task_id}: Unknown action_type '{action_type}'.")


    def add_jobs(self):
        """Adds predefined scheduled jobs."""
        self.logger.info("Adding default scheduled jobs...")

        # --- Example Agent Prompt Task ---
        self.scheduler.add_job(
            self._execute_scheduled_action,
            trigger=IntervalTrigger(minutes=30),
            id="routine_check",
            replace_existing=True,
            kwargs={ # Pass arguments via kwargs
                "action_type": "agent_prompt",
                "task_id": "routine_check",
                "prompt": "Check for any urgent items or anomalies in recent data based on available tools and knowledge."
            }
        )

        # --- Example Pipeline Task ---
        # Assumes pipeline "simple_sql_summary" is defined in PipelineHandler
        self.scheduler.add_job(
            self._execute_scheduled_action,
            trigger=CronTrigger(hour=1, minute=0, timezone="UTC"), # Daily 1 AM UTC
            id="daily_sql_summary_pipeline",
            replace_existing=True,
            kwargs={
                "action_type": "start_pipeline",
                "task_id": "daily_sql_summary_pipeline",
                "pipeline_id": "simple_sql_summary",
                # "initial_context": {} # No initial context needed for this example
            }
        )

        # --- Example Goal Task ---
        # Assumes goal type "website_monitor" is defined in GoalHandler
        self.scheduler.add_job(
            self._execute_scheduled_action,
            trigger=IntervalTrigger(hours=1), # Check every hour
            id="hourly_website_monitor_goal",
            replace_existing=True,
            kwargs={
                "action_type": "start_goal",
                "task_id": "hourly_website_monitor_goal",
                "goal_type": "website_monitor",
                "goal_description": "Perform an hourly check on the status of https://example.com, max 1 check per run." # Adjust description/params
                # Note: GoalHandler's parser needs to handle this description
            }
        )

        # --- Example Agent Prompt Task (Forecasting) ---
        self.scheduler.add_job(
            self._execute_scheduled_action,
            trigger=CronTrigger(day_of_week='mon', hour=4, minute=0, timezone="UTC"), # Monday 4 AM UTC
            id="weekly_sales_forecast_agent",
            replace_existing=True,
            kwargs={
                "action_type": "agent_prompt",
                "task_id": "weekly_sales_forecast_agent",
                "prompt": "Run the weekly sales forecast using 'sales_data.csv' and save the model and forecast data."
            }
        )

        self.logger.info(f"Scheduled jobs added: {self.scheduler.get_jobs()}")


    def start(self):
        """Adds default jobs and starts the scheduler."""
        self.add_jobs()
        if self.scheduler.get_jobs():
            try:
                self.scheduler.start()
                self.logger.info("Scheduler started successfully.")
            except Exception as e:
                 self.logger.error(f"Failed to start scheduler: {e}", exc_info=True)
        else:
            self.logger.info("Scheduler not started (no jobs configured).")

    def stop(self):
        """Stops the scheduler."""
        if self.scheduler.running:
            try:
                # wait=False allows shutdown even if jobs are running (they might get interrupted)
                self.scheduler.shutdown(wait=False)
                self.logger.info("Scheduler stopped successfully.")
            except Exception as e:
                 self.logger.error(f"Error stopping scheduler: {e}", exc_info=True)

    def get_jobs_info(self) -> List[Dict[str, Any]]:
        """Gets information about scheduled jobs."""
        # (Implementation remains the same)
        jobs_info = []
        if self.scheduler.running:
            try:
                jobs = self.scheduler.get_jobs()
                for job in jobs:
                    jobs_info.append({
                        "id": job.id,
                        "name": job.name,
                        "next_run_time": str(job.next_run_time) if job.next_run_time else "N/A",
                        "trigger": str(job.trigger)
                    })
            except Exception as e:
                 self.logger.error(f"Error getting scheduler jobs info: {e}", exc_info=True)
        return jobs_info

