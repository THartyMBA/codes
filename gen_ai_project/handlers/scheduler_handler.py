# handlers/scheduler_handler.py

import logging
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger # For cron-style scheduling

# --- Project Imports ---
from ..core.core_services import CoreAgentServices

class SchedulerHandler:
    """Manages scheduled tasks using APScheduler and CoreAgentServices."""
    def __init__(self, services: CoreAgentServices):
        self.services = services
        self.scheduler = AsyncIOScheduler(timezone="UTC") # Use UTC or your desired timezone
        self.logger = logging.getLogger(__name__)
        self.logger.info("SchedulerHandler initialized.")

    async def _run_scheduled_task(self, task_prompt: str, task_id: str):
        """The function executed by the scheduler."""
        self.logger.info(f"Running scheduled task ({task_id}): {task_prompt}")
        if not self.services.agent_executor:
            self.logger.error(f"Cannot run scheduled task {task_id}: Agent Executor not available.")
            return
        try:
            # Execute using shared agent executor
            response = await self.services.agent_executor.ainvoke({"input": task_prompt})
            output = response.get("output", "Scheduled task produced no output.")
            self.logger.info(f"Scheduled task ({task_id}) completed. Output: {output[:200]}...")
            # Log result / add to system memory via self.services
            await self.services.add_memory(f"Scheduled Task: {task_prompt}\nResult: {output}", "system_scheduler", task_id)
        except Exception as e:
            self.logger.error(f"Error executing scheduled task {task_id}: {e}", exc_info=self.services.verbose)
            await self.services.add_memory(f"Scheduled Task Failed: {task_prompt}\nError: {e}", "system_scheduler_error", task_id)

    def add_jobs(self):
        """Adds predefined scheduled jobs."""
        self.logger.info("Adding default scheduled jobs...")
        # Example 1: Interval based
        self.scheduler.add_job(
            self._run_scheduled_task,
            trigger=IntervalTrigger(minutes=30), # Run every 30 minutes
            args=["Check for any urgent items or anomalies in recent data.", "routine_check"],
            id="routine_check",
            replace_existing=True
        )
        # Example 2: Cron based (e.g., daily at 2 AM UTC)
        self.scheduler.add_job(
            self._run_scheduled_task,
            trigger=CronTrigger(hour=2, minute=0, timezone="UTC"),
            args=["Generate the daily summary report for management.", "daily_summary_report"],
            id="daily_summary_report",
            replace_existing=True
        )
        # Example 3: Run a forecast weekly
        self.scheduler.add_job(
            self._run_scheduled_task,
            trigger=CronTrigger(day_of_week='mon', hour=4, minute=0, timezone="UTC"), # Monday 4 AM UTC
            args=["Run the weekly sales forecast using 'sales_data.csv' and save the model.", "weekly_sales_forecast"],
            id="weekly_sales_forecast",
            replace_existing=True
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
                self.scheduler.shutdown()
                self.logger.info("Scheduler stopped successfully.")
            except Exception as e:
                 self.logger.error(f"Error stopping scheduler: {e}", exc_info=True)

    def get_jobs_info(self) -> List[Dict[str, Any]]:
        """Gets information about scheduled jobs."""
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

