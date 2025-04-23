# api.py (Place in the project root: gen_ai_project/)

import logging
import asyncio
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import time
import os # Needed for knowledge check

# --- Project Imports ---
from core.core_services import CoreAgentServices # Updated path
from handlers.chat_handler import ChatHandler # Updated path
from handlers.event_handler import EventHandler # Updated path
from handlers.scheduler_handler import SchedulerHandler # Updated path
from utils import config
from utils.logging_setup import setup_logging

# --- Setup ---
setup_logging() # Configure logging first
logger = logging.getLogger("api")

# --- Global Instances ---
# These will be initialized in the startup event
core_services_instance: Optional[CoreAgentServices] = None
chat_handler_instance: Optional[ChatHandler] = None
event_handler_instance: Optional[EventHandler] = None
scheduler_handler_instance: Optional[SchedulerHandler] = None
event_queue: Optional[asyncio.Queue] = None
event_processor_task: Optional[asyncio.Task] = None

# --- Request/Response Models (Keep as before, or refine) ---
class ChatRequest(BaseModel):
    user_input: str
    user_id: str = "api_user"

class ChatResponse(BaseModel):
    final_output: str
    error: Optional[str] = None
    # agent_response: Optional[Dict[str, Any]] = None # Optional: raw response

class StatusResponse(BaseModel):
    core_services_initialized: bool
    scheduler_running: bool
    scheduled_jobs: List[Dict[str, Any]]
    # running_goals: List[str] # Removed goal status for now
    event_queue_size: int

class EnqueueRequest(BaseModel):
    event_type: str
    event_data: Dict[str, Any]

# --- FastAPI App ---
app = FastAPI(title="Core Agent Services API", version="1.1")

# --- Lifespan Management ---
@app.on_event("startup")
async def startup_event():
    """Initialize Core Services and Handlers."""
    global core_services_instance, chat_handler_instance, event_handler_instance
    global scheduler_handler_instance, event_queue, event_processor_task
    logger.info("API Startup: Initializing Core Services and Handlers...")
    try:
        # 1. Initialize Core Services
        core_services_instance = CoreAgentServices(verbose=True) # Enable verbose for API debugging

        # 2. Initialize Handlers (passing services)
        chat_handler_instance = ChatHandler(core_services_instance)
        event_queue = asyncio.Queue()
        event_handler_instance = EventHandler(core_services_instance, event_queue)
        scheduler_handler_instance = SchedulerHandler(core_services_instance)

        # 3. Perform initial setup tasks (e.g., add knowledge)
        # Example: Check and add knowledge file
        knowledge_file_path = os.path.join(core_services_instance.workspace_dir, "company_policy.txt")
        if os.path.exists(knowledge_file_path):
             # Use the service's method directly
             mem_search = await core_services_instance.retrieve_memory(query="Ingested.*company_policy.txt", user_id="supervisor_system", limit=1)
             if not mem_search:
                  logger.info("API Startup: Adding initial knowledge...")
                  await core_services_instance._add_knowledge_tool_async("company_policy.txt") # Relative path
             else: logger.info("API Startup: Initial knowledge likely already present.")
        else: logger.warning("API Startup: Knowledge file 'company_policy.txt' not found in workspace.")


        # 4. Start background tasks
        scheduler_handler_instance.start() # Adds jobs and starts scheduler
        event_processor_task = asyncio.create_task(event_handler_instance.process_events()) # Start event loop

        logger.info("API Startup: Core Services and Handlers started successfully.")

    except Exception as e:
        logger.critical(f"API Startup Failed: {e}", exc_info=True)
        # Ensure instances are None if startup fails
        core_services_instance = None
        chat_handler_instance = None
        event_handler_instance = None
        scheduler_handler_instance = None
        event_queue = None
        event_processor_task = None


@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully stop Handlers."""
    logger.info("API Shutdown: Stopping Handlers...")
    if scheduler_handler_instance:
        scheduler_handler_instance.stop()
    if event_processor_task and not event_processor_task.done():
        event_processor_task.cancel()
        try:
            await event_processor_task
        except asyncio.CancelledError:
            logger.info("Event processor task cancelled successfully.")
    logger.info("API Shutdown: Handlers stopped.")

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def handle_chat_endpoint(request: ChatRequest):
    """Endpoint for interactive chat requests."""
    if not chat_handler_instance:
        raise HTTPException(status_code=503, detail="Chat Handler not available.")

    logger.info(f"Received chat request from user: {request.user_id}")
    try:
        # Delegate to the ChatHandler
        response_dict = await chat_handler_instance.handle_chat(
            user_input=request.user_input,
            user_id=request.user_id
        )
        # Adapt response if needed, here we assume handle_chat returns the desired structure
        return ChatResponse(
            final_output=response_dict.get('final_output', 'No response generated.'),
            error=response_dict.get('error'),
            # agent_response=response_dict # Optional
        )
    except Exception as e:
        logger.error(f"Error processing chat request in API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/status", response_model=StatusResponse)
async def get_status_endpoint():
    """Endpoint to get the status of background tasks."""
    if not core_services_instance or not scheduler_handler_instance or event_queue is None:
        raise HTTPException(status_code=503, detail="Core services or handlers not available.")

    logger.debug("Processing status request...")
    try:
        is_running = scheduler_handler_instance.scheduler.running if scheduler_handler_instance.scheduler else False
        jobs_info = scheduler_handler_instance.get_jobs_info()
        q_size = event_queue.qsize()

        return StatusResponse(
            core_services_initialized=True, # If we reach here, it's initialized
            scheduler_running=is_running,
            scheduled_jobs=jobs_info,
            event_queue_size=q_size
        )
    except Exception as e:
        logger.error(f"Error retrieving status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/enqueue", status_code=202) # 202 Accepted
async def enqueue_external_event_endpoint(request: EnqueueRequest):
    """Endpoint for external systems to enqueue events."""
    if not event_queue: # Check if the queue exists
        raise HTTPException(status_code=503, detail="Event queue not available.")

    logger.info(f"Received external event to enqueue via API: Type='{request.event_type}'")
    try:
        await event_queue.put((request.event_type, request.event_data))
        return {"message": "Event enqueued successfully."}
    except Exception as e:
        logger.error(f"Error enqueuing event via API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Main block to run API (if run directly) ---
if __name__ == "__main__":
    logger.info("Starting API server directly using uvicorn...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False, log_level="info")

    # Note: In production, consider using a WSGI server like Gunicorn with Uvicorn workers.
    # Example: gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --host