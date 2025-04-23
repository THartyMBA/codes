# api.py (Place in the project root: gen_ai_project/)

import logging
import asyncio
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import time

# --- Project Imports ---
# Assuming this runs from the project root
from agents.supervisor_agent import SupervisorAgent
from utils import config
from utils.logging_setup import setup_logging

# --- Setup ---
setup_logging() # Configure logging first
logger = logging.getLogger("api")

# --- Global Supervisor Instance ---
# This instance will be shared across API requests
# Initialized within the lifespan event handler
supervisor_instance: Optional[SupervisorAgent] = None

# --- Request/Response Models ---
class ChatRequest(BaseModel):
    user_input: str
    user_id: str = "streamlit_user" # Default user ID for Streamlit sessions

class ChatResponse(BaseModel):
    final_output: str
    error: Optional[str] = None
    agent_response: Optional[Dict[str, Any]] = None # Include raw agent response if needed

class StatusResponse(BaseModel):
    scheduler_running: bool
    scheduled_jobs: List[Dict[str, Any]]
    running_goals: List[str] # List of running goal IDs
    event_queue_size: int

class EnqueueRequest(BaseModel):
    event_type: str
    event_data: Dict[str, Any]

# --- FastAPI App ---
app = FastAPI(title="Supervisor Agent API", version="1.0")

# --- Lifespan Management (Initialize/Shutdown Supervisor) ---
@app.on_event("startup")
async def startup_event():
    """Initialize the SupervisorAgent and start its background tasks."""
    global supervisor_instance
    logger.info("API Startup: Initializing SupervisorAgent...")
    try:
        supervisor_instance = SupervisorAgent(
            model_name=config.OLLAMA_MODEL,
            temperature=config.DEFAULT_LLM_TEMPERATURE,
            db_uri=config.DB_URI,
            verbose=True # Enable verbose logging for debugging API calls
        )
        # Add initial knowledge if needed (or ensure it's done elsewhere)
        # Example: Check and add knowledge file
        knowledge_file_path = os.path.join(supervisor_instance.workspace_dir, "company_policy.txt")
        if os.path.exists(knowledge_file_path):
             mem_search = await asyncio.to_thread(supervisor_instance.mem0_memory.search, query="Ingested.*company_policy.txt", user_id="supervisor_system", limit=1)
             if not mem_search:
                  logger.info("API Startup: Adding initial knowledge...")
                  await supervisor_instance._add_knowledge_tool_async("company_policy.txt") # Relative path to workspace
             else: logger.info("API Startup: Initial knowledge likely already present.")

        # Add scheduled tasks (example)
        logger.info("API Startup: Adding scheduled tasks...")
        supervisor_instance.add_scheduled_task(
            task_func=supervisor_instance._arun_autonomous_task,
            trigger='interval',
            task_id="api_scheduled_summary",
            task_prompt="Provide a brief summary of recent activities or database state.",
            minutes=15 # Example: Run every 15 minutes
        )

        await supervisor_instance.start() # Start scheduler & event queue processor
        logger.info("API Startup: SupervisorAgent started successfully.")
    except Exception as e:
        logger.critical(f"API Startup Failed: Could not initialize/start SupervisorAgent: {e}", exc_info=True)
        # Optionally raise error to prevent API from starting
        supervisor_instance = None # Ensure it's None if startup failed

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully stop the SupervisorAgent."""
    logger.info("API Shutdown: Stopping SupervisorAgent...")
    if supervisor_instance:
        await supervisor_instance.stop()
    logger.info("API Shutdown: SupervisorAgent stopped.")

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """Endpoint for interactive chat requests."""
    if not supervisor_instance:
        raise HTTPException(status_code=503, detail="Supervisor Agent not available.")

    logger.info(f"Received chat request from user: {request.user_id}")
    try:
        response_dict = await supervisor_instance.arun_interactive(
            user_input=request.user_input,
            user_id=request.user_id
        )
        return ChatResponse(
            final_output=response_dict.get('final_output', 'No response generated.'),
            error=response_dict.get('error'),
            agent_response=response_dict # Send back the full dict if needed by UI
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Endpoint to get the status of background tasks."""
    if not supervisor_instance:
        raise HTTPException(status_code=503, detail="Supervisor Agent not available.")

    logger.debug("Processing status request...")
    try:
        jobs_info = []
        if supervisor_instance.scheduler.running:
            jobs = supervisor_instance.scheduler.get_jobs()
            for job in jobs:
                jobs_info.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": str(job.next_run_time) if job.next_run_time else "N/A",
                    "trigger": str(job.trigger)
                })

        running_goal_ids = list(supervisor_instance.running_goals.keys())
        queue_size = supervisor_instance.event_queue.qsize()

        return StatusResponse(
            scheduler_running=supervisor_instance.scheduler.running,
            scheduled_jobs=jobs_info,
            running_goals=running_goal_ids,
            event_queue_size=queue_size
        )
    except Exception as e:
        logger.error(f"Error retrieving status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/enqueue", status_code=202) # 202 Accepted
async def enqueue_external_event(request: EnqueueRequest):
    """Endpoint for external systems to enqueue events."""
    if not supervisor_instance:
        raise HTTPException(status_code=503, detail="Supervisor Agent not available.")

    logger.info(f"Received external event to enqueue: Type='{request.event_type}'")
    try:
        await supervisor_instance.enqueue_event(request.event_type, request.event_data)
        return {"message": "Event enqueued successfully."}
    except Exception as e:
        logger.error(f"Error enqueuing event: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Main block to run API (for direct execution) ---
# Typically, you'd use `run.py` or docker to manage this process
if __name__ == "__main__":
    logger.info("Starting API server directly...")
    # Use host="0.0.0.0" to make it accessible on the network
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
    # Note: `reload=True` is useful for development but might interfere with the supervisor's state.
