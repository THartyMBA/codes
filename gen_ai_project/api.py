# api.py (Place in the project root: gen_ai_project/)

import logging
import asyncio
from fastapi import FastAPI, HTTPException, Body, Depends, Security # Added Security
from fastapi.security import APIKeyHeader # Import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uvicorn
import time
import os
import json

# --- Project Imports ---
from core.core_services import CoreAgentServices
from handlers.chat_handler import ChatHandler
from handlers.event_handler import EventHandler
from handlers.scheduler_handler import SchedulerHandler
from handlers.goal_handler import GoalHandler
from handlers.pipeline_handler import PipelineHandler
from handlers.monitoring_handler import MonitoringHandler
from handlers.knowledge_handler import KnowledgeHandler
from handlers.admin_handler import AdminHandler
from utils import config # Import config to access ADMIN_API_KEY
from utils.logging_setup import setup_logging

# --- Setup ---
setup_logging()
logger = logging.getLogger("api")

# --- Global Instances ---
# These will be initialized in the startup event
core_services_instance: Optional[CoreAgentServices] = None
chat_handler_instance: Optional[ChatHandler] = None
event_handler_instance: Optional[EventHandler] = None
scheduler_handler_instance: Optional[SchedulerHandler] = None
goal_handler_instance: Optional[GoalHandler] = None
pipeline_handler_instance: Optional[PipelineHandler] = None
monitoring_handler_instance: Optional[MonitoringHandler] = None
knowledge_handler_instance: Optional[KnowledgeHandler] = None
admin_handler_instance: Optional[AdminHandler] = None
event_queue: Optional[asyncio.Queue] = None
# event_processor_task is managed internally by EventHandler now

# --- API Key Security Definition ---
API_KEY_NAME = "X-API-Key" # Define header name
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key_header: Optional[str] = Security(api_key_header_auth)):
    """Dependency function to verify the provided API key."""
    if not config.ADMIN_API_KEY:
        logger.error("Admin API Key is not configured on the server.")
        raise HTTPException(status_code=500, detail="Admin actions are disabled due to missing server configuration.")

    if api_key_header is None:
        logger.warning("Admin endpoint access attempt failed: Missing API Key header.")
        raise HTTPException(
            status_code=403, detail=f"Missing API Key header: '{API_KEY_NAME}'"
        )
    if api_key_header != config.ADMIN_API_KEY:
        logger.warning("Admin endpoint access attempt failed: Invalid API Key.")
        raise HTTPException(
            status_code=403, detail="Invalid API Key"
        )
    # If key is valid, proceed
    return api_key_header # Return the key or True

# --- Request/Response Models ---
# (Keep existing Pydantic models as defined previously)
# Chat
class ChatRequest(BaseModel):
    user_input: str
    user_id: str = "api_user"

class ChatResponse(BaseModel):
    final_output: str
    error: Optional[str] = None

# Status
class BasicStatusResponse(BaseModel):
    status: str = "OK"
    timestamp: float = Field(default_factory=time.time)

# Enqueue Event
class EnqueueRequest(BaseModel):
    event_type: str
    event_data: Dict[str, Any]

# Goal Management
class StartGoalRequest(BaseModel):
    goal_description: str
    goal_type: str

class GoalResponse(BaseModel):
    goal_id: Optional[str] = None
    status: str
    error: Optional[str] = None
    state: Optional[Dict[str, Any]] = None
    state_snapshot: Optional[Dict[str, Any]] = None

# Pipeline Management
class StartPipelineRequest(BaseModel):
    pipeline_id: str
    initial_context: Optional[Dict[str, Any]] = None

class PipelineResponse(BaseModel):
    run_id: Optional[str] = None
    status: str
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

# Admin Actions
class KnowledgeReindexRequest(BaseModel):
    source_path: str
    force_add: bool = False

class AdminActionResponse(BaseModel):
    status: str
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# --- FastAPI App ---
app = FastAPI(
    title="GenAI Agent System API",
    version="1.2",
    description="API for interacting with Core Agent Services and Handlers"
)

# --- Lifespan Management ---
@app.on_event("startup")
async def startup_event():
    """Initialize Core Services and Handlers."""
    global core_services_instance, chat_handler_instance, event_handler_instance
    global scheduler_handler_instance, goal_handler_instance, pipeline_handler_instance
    global monitoring_handler_instance, knowledge_handler_instance, admin_handler_instance
    global event_queue # Removed event_processor_task global

    logger.info("API Startup: Initializing Core Services and Handlers...")

    # --- Check for Admin API Key ---
    if not config.ADMIN_API_KEY:
        logger.critical("CRITICAL: ADMIN_API_KEY is not set in the environment/config. Admin endpoints will be inaccessible.")
    else:
        logger.info("Admin API Key found in configuration.")
    # --- End Check ---

    try:
        # 1. Initialize Event Queue first
        event_queue = asyncio.Queue()

        # 2. Initialize Core Services (Pass event queue)
        # CoreServices now initializes Knowledge, Goal, Pipeline, Monitoring, Admin handlers internally
        core_services_instance = CoreAgentServices(event_queue=event_queue, verbose=True)

        # 3. Initialize Handlers managed at API level (Chat, Event, Scheduler)
        chat_handler_instance = ChatHandler(core_services_instance)
        event_handler_instance = EventHandler(core_services_instance, event_queue)
        scheduler_handler_instance = SchedulerHandler(core_services_instance)

        # 4. Assign references to externally managed handlers for AdminHandler
        # Access internally managed handlers via core_services_instance if needed by AdminHandler logic
        if core_services_instance.admin_handler:
             core_services_instance.admin_handler.ext_scheduler_handler = scheduler_handler_instance
             core_services_instance.admin_handler.ext_event_handler = event_handler_instance
             logger.info("Provided external handler references to AdminHandler.")

        # 5. Perform initial setup tasks (e.g., add knowledge via KnowledgeHandler)
        knowledge_handler_instance = core_services_instance.knowledge_handler # Get ref
        if knowledge_handler_instance:
            knowledge_file_path = os.path.join(core_services_instance.workspace_dir, "company_policy.txt")
            if os.path.exists(knowledge_file_path):
                mem_search = await core_services_instance.retrieve_memory(query="Ingested.*company_policy.txt", user_id="system_knowledge", limit=1)
                if not mem_search:
                    logger.info("API Startup: Adding initial knowledge via KnowledgeHandler...")
                    await knowledge_handler_instance.add_source("company_policy.txt")
                else: logger.info("API Startup: Initial knowledge likely already present.")
            else: logger.warning("API Startup: Knowledge file 'company_policy.txt' not found in workspace.")
        else: logger.error("Knowledge Handler not initialized in Core Services.")


        # 6. Start background tasks managed by handlers
        scheduler_handler_instance.start()
        # Access monitoring handler via core services to start it
        if core_services_instance.monitoring_handler:
             core_services_instance.monitoring_handler.start()
        else: logger.error("Monitoring Handler not initialized in Core Services.")
        # Start event handler's loop via its own method
        event_handler_instance.start()

        logger.info("API Startup: Core Services and Handlers started successfully.")

    except Exception as e:
        logger.critical(f"API Startup Failed: {e}", exc_info=True)
        # Ensure instances are None if startup fails
        core_services_instance = None
        chat_handler_instance = None
        event_handler_instance = None
        scheduler_handler_instance = None
        # Add None assignments for internally managed handlers if needed for clarity
        goal_handler_instance = None
        pipeline_handler_instance = None
        monitoring_handler_instance = None
        knowledge_handler_instance = None
        admin_handler_instance = None
        event_queue = None


@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully stop Handlers and background tasks."""
    logger.info("API Shutdown: Stopping Handlers and background tasks...")
    # Stop handlers in a reasonable order
    # Access internally managed handlers via core_services_instance
    if core_services_instance and core_services_instance.monitoring_handler:
        await core_services_instance.monitoring_handler.stop()
    if scheduler_handler_instance: # Externally managed
        scheduler_handler_instance.stop() # Usually sync stop

    # Goal/Pipeline stop methods should handle cancelling their running tasks
    if core_services_instance and core_services_instance.goal_handler and hasattr(core_services_instance.goal_handler, 'stop'):
        await core_services_instance.goal_handler.stop()
    if core_services_instance and core_services_instance.pipeline_handler:
        await core_services_instance.pipeline_handler.stop()

    # Stop Admin/Knowledge handlers if they have stop logic
    if core_services_instance and core_services_instance.admin_handler and hasattr(core_services_instance.admin_handler, 'stop'):
        await core_services_instance.admin_handler.stop()
    if core_services_instance and core_services_instance.knowledge_handler and hasattr(core_services_instance.knowledge_handler, 'stop'):
        await core_services_instance.knowledge_handler.stop()

    # Stop the event handler's loop via its method (externally managed)
    if event_handler_instance:
        await event_handler_instance.stop()

    logger.info("API Shutdown: Handlers stopped.")

# --- Helper Function ---
async def get_chat_handler() -> ChatHandler:
    if not chat_handler_instance: raise HTTPException(status_code=503, detail="Chat Handler not available.")
    return chat_handler_instance

# --- API Endpoints ---

# Basic Status (Public)
@app.get("/status", response_model=BasicStatusResponse, tags=["Status"])
async def get_basic_status():
    if not core_services_instance: raise HTTPException(status_code=503, detail="Core Services not initialized.")
    return BasicStatusResponse(status="OK")

# Chat (Public)
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def handle_chat_endpoint(request: ChatRequest, handler: ChatHandler = Depends(get_chat_handler)):
    logger.info(f"Received chat request from user: {request.user_id}")
    try:
        response_dict = await handler.handle_chat(request.user_input, request.user_id)
        return ChatResponse(
            final_output=response_dict.get('final_output', 'No response generated.'),
            error=response_dict.get('error')
        )
    except Exception as e:
        logger.error(f"Error processing chat request in API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# Event Enqueue (Potentially needs security depending on source)
@app.post("/events", status_code=202, tags=["Events"])
async def enqueue_external_event_endpoint(request: EnqueueRequest):
    if not event_queue: raise HTTPException(status_code=503, detail="Event queue not available.")
    logger.info(f"Received external event to enqueue via API: Type='{request.event_type}'")
    try:
        await event_queue.put((request.event_type, request.event_data))
        return {"message": "Event enqueued successfully."}
    except Exception as e:
        logger.error(f"Error enqueuing event via API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# Goal Management (Potentially needs security)
@app.post("/goals", response_model=GoalResponse, status_code=201, tags=["Goals"])
async def start_new_goal_endpoint(request: StartGoalRequest):
    # Access goal handler via core services
    if not core_services_instance or not core_services_instance.goal_handler:
        raise HTTPException(status_code=503, detail="Goal Handler not available.")
    try:
        result = await core_services_instance.goal_handler.start_goal(request.goal_description, request.goal_type)
        if "error" in result:
             status_code = 400 if "parse" in result["error"].lower() or "type" in result["error"].lower() else 500
             raise HTTPException(status_code=status_code, detail=result["error"])
        return GoalResponse(**result)
    except Exception as e: logger.error(f"Error in /goals endpoint: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/goals/{goal_id}/status", response_model=GoalResponse, tags=["Goals"])
async def get_goal_status_endpoint(goal_id: str):
    if not core_services_instance or not core_services_instance.goal_handler:
        raise HTTPException(status_code=503, detail="Goal Handler not available.")
    try:
        result = await core_services_instance.goal_handler.get_goal_status(goal_id)
        if result["status"] == "not_found": raise HTTPException(status_code=404, detail=f"Goal ID {goal_id} not found.")
        return GoalResponse(**result)
    except HTTPException: raise
    except Exception as e: logger.error(f"Error getting status for goal {goal_id}: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.delete("/goals/{goal_id}", response_model=GoalResponse, tags=["Goals"])
async def cancel_existing_goal_endpoint(goal_id: str):
    if not core_services_instance or not core_services_instance.goal_handler:
        raise HTTPException(status_code=503, detail="Goal Handler not available.")
    try:
        result = await core_services_instance.goal_handler.cancel_goal(goal_id)
        if result["status"] == "not_found": raise HTTPException(status_code=404, detail=f"Goal ID {goal_id} not found.")
        return GoalResponse(**result)
    except HTTPException: raise
    except Exception as e: logger.error(f"Error cancelling goal {goal_id}: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# Pipeline Management (Potentially needs security)
@app.post("/pipelines/{pipeline_id}/run", response_model=PipelineResponse, status_code=201, tags=["Pipelines"])
async def start_new_pipeline_endpoint(pipeline_id: str, request: StartPipelineRequest):
    # Access pipeline handler via core services
    if not core_services_instance or not core_services_instance.pipeline_handler:
        raise HTTPException(status_code=503, detail="Pipeline Handler not available.")
    if pipeline_id != request.pipeline_id: raise HTTPException(status_code=400, detail="Pipeline ID in path does not match request body.")
    try:
        result = await core_services_instance.pipeline_handler.start_pipeline(request.pipeline_id, request.initial_context)
        if "error" in result:
             status_code = 404 if "not found" in result["error"].lower() else 500
             raise HTTPException(status_code=status_code, detail=result["error"])
        return PipelineResponse(**result)
    except Exception as e: logger.error(f"Error starting pipeline {pipeline_id}: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/pipelines/runs/{run_id}", response_model=PipelineResponse, tags=["Pipelines"])
async def get_pipeline_status_endpoint(run_id: str):
    if not core_services_instance or not core_services_instance.pipeline_handler:
        raise HTTPException(status_code=503, detail="Pipeline Handler not available.")
    try:
        result = core_services_instance.pipeline_handler.get_pipeline_status(run_id)
        if result["status"] == "not_found": raise HTTPException(status_code=404, detail=f"Pipeline run ID {run_id} not found.")
        return PipelineResponse(**result)
    except HTTPException: raise
    except Exception as e: logger.error(f"Error getting status for pipeline run {run_id}: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.delete("/pipelines/runs/{run_id}", response_model=PipelineResponse, tags=["Pipelines"])
async def cancel_pipeline_run_endpoint(run_id: str):
    if not core_services_instance or not core_services_instance.pipeline_handler:
        raise HTTPException(status_code=503, detail="Pipeline Handler not available.")
    try:
        result = await core_services_instance.pipeline_handler.cancel_pipeline(run_id)
        if result["status"] == "not_found": raise HTTPException(status_code=404, detail=f"Pipeline run ID {run_id} not found.")
        return PipelineResponse(**result)
    except HTTPException: raise
    except Exception as e: logger.error(f"Error cancelling pipeline run {run_id}: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- Admin Endpoints (Applying Security) ---
admin_dependencies = [Depends(verify_api_key)] # Define the dependency list

@app.get("/admin/overview", response_model=AdminActionResponse, tags=["Admin"], dependencies=admin_dependencies)
async def get_admin_overview():
    """Get a high-level overview of the system status. Requires Admin API Key."""
    # Access admin handler via core services
    if not core_services_instance or not core_services_instance.admin_handler:
        raise HTTPException(status_code=503, detail="Admin Handler not available.")
    try:
        overview_data = await core_services_instance.admin_handler.get_system_overview()
        return AdminActionResponse(status="success", details=overview_data)
    except Exception as e: logger.error(f"Error getting admin overview: {e}", exc_info=True); raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/health", response_model=AdminActionResponse, tags=["Admin"], dependencies=admin_dependencies)
async def get_admin_health():
    """Perform basic health checks on core components. Requires Admin API Key."""
    if not core_services_instance or not core_services_instance.admin_handler:
        raise HTTPException(status_code=503, detail="Admin Handler not available.")
    try:
        health_data = await core_services_instance.admin_handler.check_system_health()
        # Return 200 even if unhealthy, but indicate status in response body
        return AdminActionResponse(status=health_data.get("overall_status", "UNKNOWN"), details=health_data)
    except Exception as e: logger.error(f"Error getting system health: {e}", exc_info=True); raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/knowledge/reindex", response_model=AdminActionResponse, tags=["Admin"], dependencies=admin_dependencies)
async def trigger_admin_knowledge_reindex(request: KnowledgeReindexRequest):
    """Trigger knowledge ingestion for a specific source. Requires Admin API Key."""
    if not core_services_instance or not core_services_instance.admin_handler:
        raise HTTPException(status_code=503, detail="Admin Handler not available.")
    try:
        # Admin handler calls knowledge handler internally
        result = await core_services_instance.admin_handler.trigger_knowledge_reindex(request.source_path, request.force_add)
        return AdminActionResponse(**result)
    except Exception as e: logger.error(f"Error triggering knowledge reindex: {e}", exc_info=True); raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/memory/{user_id}", response_model=AdminActionResponse, tags=["Admin"], dependencies=admin_dependencies)
async def clear_admin_user_memory(user_id: str):
    """(Experimental) Clear conversation memory for a user. Requires Admin API Key."""
    if not core_services_instance or not core_services_instance.admin_handler:
        raise HTTPException(status_code=503, detail="Admin Handler not available.")
    try:
        result = await core_services_instance.admin_handler.clear_user_memory(user_id)
        return AdminActionResponse(**result)
    except Exception as e: logger.error(f"Error clearing user memory: {e}", exc_info=True); raise HTTPException(status_code=500, detail=str(e))

# --- Main block ---
if __name__ == "__main__":
    logger.info("Starting API server directly using uvicorn...")
    # Ensure necessary directories exist before starting (moved from config for clarity)
    os.makedirs(config.WORKSPACE_DIR, exist_ok=True)
    os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
    os.makedirs(config.CONFIG_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.WORKSPACE_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(config.WORKSPACE_DIR, "autonomous_logs"), exist_ok=True)
    os.makedirs(os.path.join(config.WORKSPACE_DIR, "templates"), exist_ok=True) # For reporting

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False, log_level="info")

    # Note: In production, consider using a WSGI server like Gunicorn or Uvicorn with ASGI for better performance.
    # For example, use: gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --bind