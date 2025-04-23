# graphs/website_monitor.py

import logging
import asyncio
from typing import TypedDict, Annotated, Optional, Sequence
import operator
import time

import httpx # Async HTTP client (pip install httpx)
from langgraph.graph import StateGraph, END, CompiledGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)

# --- State Definition ---

class WebsiteMonitorState(TypedDict):
    """
    Represents the state of the website monitoring goal.

    Attributes:
        target_url: The URL to monitor.
        max_checks: Maximum number of checks to perform if the site is OK.
        check_interval_seconds: How long to wait between checks if the site is OK.
        current_check_count: Counter for checks performed in this run.
        last_status_code: HTTP status code from the last check.
        last_check_time: Timestamp of the last check.
        error_message: Error message if the check itself failed (e.g., connection error).
        alert_message: Final message generated if the website is down or check failed repeatedly.
    """
    target_url: str
    max_checks: int
    check_interval_seconds: int = 60 # Default wait time
    current_check_count: Annotated[int, operator.add] # Use operator.add if accumulating over runs, otherwise just int
    last_status_code: Optional[int]
    last_check_time: Optional[float]
    error_message: Optional[str]
    alert_message: Optional[str]


# --- Node Functions (Async) ---

async def increment_counter(state: WebsiteMonitorState) -> Dict[str, Any]:
    """Increments the check counter."""
    count = state.get("current_check_count", 0) + 1
    logger.info(f"WebsiteMonitor ({state.get('target_url', 'N/A')}): Check #{count}")
    return {"current_check_count": count}

async def check_website(state: WebsiteMonitorState) -> Dict[str, Any]:
    """Performs the HTTP check on the target URL."""
    url = state.get("target_url")
    if not url:
        logger.error("WebsiteMonitor: Target URL is missing in state.")
        return {"error_message": "Target URL not provided."}

    logger.debug(f"WebsiteMonitor: Checking URL: {url}")
    status_code = None
    error_msg = None
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client: # Added timeout and redirects
            response = await client.get(url)
            status_code = response.status_code
            logger.info(f"WebsiteMonitor: Check for {url} completed. Status Code: {status_code}")
            if status_code >= 400: # Treat 4xx and 5xx as errors for monitoring
                 error_msg = f"Received non-success status code: {status_code}"
                 logger.warning(f"WebsiteMonitor: {error_msg} for {url}")

    except httpx.TimeoutException:
        error_msg = "Request timed out."
        logger.warning(f"WebsiteMonitor: {error_msg} for {url}")
    except httpx.RequestError as e:
        error_msg = f"HTTP request error: {e.__class__.__name__}"
        logger.error(f"WebsiteMonitor: {error_msg} checking {url}: {e}")
    except Exception as e:
        error_msg = f"An unexpected error occurred during check: {e}"
        logger.error(f"WebsiteMonitor: {error_msg} checking {url}", exc_info=True)

    return {
        "last_status_code": status_code,
        "last_check_time": time.time(),
        "error_message": error_msg # Will be None if check was successful (status < 400)
    }

async def generate_alert(state: WebsiteMonitorState) -> Dict[str, Any]:
    """Generates an alert message if the check failed."""
    url = state.get("target_url", "N/A")
    status = state.get("last_status_code", "N/A")
    check_error = state.get("error_message")
    count = state.get("current_check_count", 0)

    if check_error:
        message = f"ALERT: Website check for {url} failed after {count} attempt(s). Error: {check_error}"
    else:
        # This node is only reached if status code indicated an issue
        message = f"ALERT: Website {url} is responding with status {status} (attempt {count})."

    logger.warning(f"WebsiteMonitor: Generating Alert: {message}")
    return {"alert_message": message}

async def wait_step(state: WebsiteMonitorState) -> Dict[str, Any]:
    """Waits for the specified interval before the next check."""
    interval = state.get("check_interval_seconds", 60)
    logger.debug(f"WebsiteMonitor: Waiting for {interval} seconds...")
    await asyncio.sleep(interval)
    logger.debug("WebsiteMonitor: Wait finished.")
    return {} # No state change needed


# --- Conditional Edge Logic ---

def should_continue_or_alert(state: WebsiteMonitorState) -> str:
    """Determines the next step after checking the website."""
    check_error = state.get("error_message")
    status_code = state.get("last_status_code")
    current_count = state.get("current_check_count", 0)
    max_checks = state.get("max_checks", 3)

    # Check if the website check itself failed or returned an error status code
    if check_error or (status_code is not None and status_code >= 400):
        logger.warning(f"WebsiteMonitor: Condition met for alert (Error: {check_error}, Status: {status_code}).")
        return "generate_alert_node" # Route to generate alert

    # If check was OK, decide whether to continue checking or end
    if current_count < max_checks:
        logger.debug(f"WebsiteMonitor: Check OK, count ({current_count}) < max ({max_checks}). Continuing.")
        return "wait_node" # Route to wait before next check cycle
    else:
        logger.info(f"WebsiteMonitor: Check OK and reached max checks ({max_checks}). Ending goal.")
        return END # End the graph execution


# --- Graph Builder Function ---

def get_website_monitor_app(checkpointer: BaseCheckpointSaver) -> CompiledGraph:
    """Builds and compiles the LangGraph application for website monitoring."""
    logger.debug("Building Website Monitor Graph...")

    graph = StateGraph(WebsiteMonitorState)

    # Add nodes
    graph.add_node("increment_counter_node", increment_counter)
    graph.add_node("check_website_node", check_website)
    graph.add_node("generate_alert_node", generate_alert)
    graph.add_node("wait_node", wait_step)

    # Define edges
    graph.set_entry_point("increment_counter_node")
    graph.add_edge("increment_counter_node", "check_website_node")

    # Conditional edge after checking the website
    graph.add_conditional_edges(
        "check_website_node", # Source node
        should_continue_or_alert, # Function to decide the route
        {
            "generate_alert_node": "generate_alert_node", # Map return value to node name
            "wait_node": "wait_node",
            END: END
        }
    )

    # Edge after generating an alert (end the process)
    graph.add_edge("generate_alert_node", END)

    # Edge after waiting (loop back to increment counter for next check)
    graph.add_edge("wait_node", "increment_counter_node")

    # Compile the graph with the checkpointer
    app = graph.compile(checkpointer=checkpointer)
    logger.debug("Website Monitor Graph compiled.")
    return app

# --- Placeholder for other graph types ---
# def get_data_pipeline_app(checkpointer: BaseCheckpointSaver) -> CompiledGraph:
#     # Define State, Nodes, Edges for data pipeline graph here...
#     logger.warning("Data Pipeline Graph not implemented.")
#     return None

