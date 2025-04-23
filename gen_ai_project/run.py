# run.py (Place in the project root: gen_ai_project/)

import asyncio
import subprocess
import sys
import signal
import logging
import uvicorn
import os

# --- Project Imports ---
from utils.logging_setup import setup_logging
from api import app # Import the FastAPI app instance from api.py

# --- Setup ---
setup_logging() # Configure logging first
logger = logging.getLogger("run")

# --- Configuration ---
API_HOST = "127.0.0.1"
API_PORT = 8000
STREAMLIT_PORT = 8501

# --- Global Process Handles ---
streamlit_process = None

# --- Signal Handling for Graceful Shutdown ---
async def shutdown(sig, loop):
    logger.info(f"Received exit signal {sig.name}...")
    global streamlit_process

    # Terminate Streamlit process
    if streamlit_process and streamlit_process.poll() is None: # Check if running
        logger.info("Terminating Streamlit process...")
        streamlit_process.terminate()
        try:
            # Wait briefly for termination
            streamlit_process.wait(timeout=5)
            logger.info("Streamlit process terminated.")
        except subprocess.TimeoutExpired:
            logger.warning("Streamlit process did not terminate gracefully, killing...")
            streamlit_process.kill()

    # FastAPI/Uvicorn usually handles SIGINT/SIGTERM itself via Uvicorn's signal handling
    # The API's shutdown event handler will stop the supervisor.
    # We might need to stop the asyncio loop more explicitly if needed.
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    logger.info(f"Cancelling {len(tasks)} outstanding asyncio tasks.")
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


async def main():
    loop = asyncio.get_running_loop()

    # Add signal handlers for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s, loop)))

    # --- Start FastAPI/Uvicorn Server ---
    # Uvicorn needs to be run in a way that integrates with the current asyncio loop
    # Running it directly via uvicorn.run() blocks. We use uvicorn.Server.
    config = uvicorn.Config("api:app", host=API_HOST, port=API_PORT, log_level="info", lifespan="on")
    server = uvicorn.Server(config)
    # Run the server in the background using loop.create_task
    uvicorn_task = loop.create_task(server.serve())
    logger.info(f"FastAPI server starting on http://{API_HOST}:{API_PORT}")
    # Give Uvicorn a moment to start up before launching Streamlit
    await asyncio.sleep(2)

    # --- Launch Streamlit App ---
    global streamlit_process
    streamlit_cmd = [
        sys.executable, # Use the same python interpreter
        "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.address", API_HOST, # Use same host for consistency
        "--server.headless", "true", # Prevents opening browser automatically
        "--browser.gatherUsageStats", "false"
    ]
    logger.info(f"Launching Streamlit: {' '.join(streamlit_cmd)}")
    try:
        # Use Popen for non-blocking execution
        streamlit_process = subprocess.Popen(streamlit_cmd)
        logger.info(f"Streamlit app launched on http://{API_HOST}:{STREAMLIT_PORT}")
    except Exception as e:
        logger.critical(f"Failed to launch Streamlit: {e}", exc_info=True)
        # Optionally stop the API server if Streamlit fails to launch
        server.should_exit = True # Signal uvicorn to stop
        await uvicorn_task # Wait for uvicorn task to finish
        return # Exit

    # Keep the main function alive while servers are running
    # Wait for the Uvicorn task to complete (which happens on shutdown)
    await uvicorn_task

    # Final check on Streamlit process after Uvicorn stops
    if streamlit_process and streamlit_process.poll() is None:
        logger.info("Uvicorn stopped, ensuring Streamlit is also terminated.")
        streamlit_process.terminate()
        try: streamlit_process.wait(timeout=2)
        except subprocess.TimeoutExpired: streamlit_process.kill()

    logger.info("Run script finished.")


if __name__ == "__main__":
    # Ensure workspace and log directories exist before starting
    os.makedirs(config.WORKSPACE_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.WORKSPACE_DIR, "logs"), exist_ok=True) # For logging_setup default
    os.makedirs(os.path.join(config.WORKSPACE_DIR, "autonomous_logs"), exist_ok=True) # For supervisor logs

    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Main execution loop interrupted or exited.")

