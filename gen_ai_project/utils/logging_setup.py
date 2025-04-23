# utils/logging_setup.py

import logging
import logging.handlers
import os
import sys

# Import config to potentially get log path or level settings
# Use a try-except block in case config itself has issues during early setup
try:
    from . import config
    # Determine log directory (e.g., inside workspace or a dedicated logs dir)
    _default_log_dir = os.path.join(config.WORKSPACE_DIR, "logs")
    LOG_DIRECTORY = os.getenv("LOG_DIRECTORY", _default_log_dir)
    LOG_FILENAME = os.getenv("LOG_FILENAME", "gen_ai_app.log")
    LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, LOG_FILENAME)
except ImportError:
    print("Warning [Logging Setup]: Could not import config. Using default log path './logs/gen_ai_app.log'.")
    LOG_DIRECTORY = "./logs"
    LOG_FILENAME = "gen_ai_app.log"
    LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, LOG_FILENAME)
except Exception as e:
    print(f"Warning [Logging Setup]: Error accessing config for log path: {e}. Using default './logs/gen_ai_app.log'.")
    LOG_DIRECTORY = "./logs"
    LOG_FILENAME = "gen_ai_app.log"
    LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, LOG_FILENAME)


# --- Configuration ---
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
# Can be set via environment variable LOG_LEVEL
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()

# Log format
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# File rotation settings
# when: 'S' Seconds, 'M' Minutes, 'H' Hours, 'D' Days, 'W0'-'W6' Weekday (0=Monday), 'midnight'
LOG_ROTATION_WHEN = "midnight"
LOG_ROTATION_INTERVAL = 1 # Interval based on 'when' (e.g., 1 day if when='midnight')
LOG_ROTATION_BACKUP_COUNT = 7 # Number of backup files to keep


def setup_logging():
    """
    Configures logging for the application.
    Sets up console and rotating file handlers.
    Should be called once at the application start.
    """
    log_level = getattr(logging, LOG_LEVEL_STR, logging.INFO)

    # --- Create Log Directory ---
    try:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
    except OSError as e:
        print(f"Error [Logging Setup]: Could not create log directory '{LOG_DIRECTORY}': {e}", file=sys.stderr)
        # Optionally fall back to logging only to console or raise error
        # For now, we'll proceed, FileHandler will likely fail later

    # --- Create Formatter ---
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # --- Create Handlers ---
    # Console Handler (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level) # Console shows messages at specified level or higher
    console_handler.setFormatter(formatter)

    # Rotating File Handler
    try:
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=LOG_FILE_PATH,
            when=LOG_ROTATION_WHEN,
            interval=LOG_ROTATION_INTERVAL,
            backupCount=LOG_ROTATION_BACKUP_COUNT,
            encoding='utf-8'
        )
        # File handler can optionally log more detail (e.g., DEBUG)
        # file_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(log_level) # Or keep same as console
        file_handler.setFormatter(formatter)
        file_handler_active = True
    except Exception as e:
        print(f"Error [Logging Setup]: Failed to create file handler for '{LOG_FILE_PATH}': {e}", file=sys.stderr)
        file_handler_active = False


    # --- Configure Root Logger ---
    # Get the root logger
    root_logger = logging.getLogger()

    # Set the root logger's level. This is the lowest level it will process.
    root_logger.setLevel(log_level)

    # Remove existing handlers (important to avoid duplicates if setup is called again)
    # This is crucial especially in interactive environments like Jupyter
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add the handlers
    root_logger.addHandler(console_handler)
    if file_handler_active:
        root_logger.addHandler(file_handler)

    # --- Initial Log Message ---
    initial_message = f"Logging configured. Level: {LOG_LEVEL_STR}."
    if file_handler_active:
        initial_message += f" Log file: '{LOG_FILE_PATH}'"
    else:
        initial_message += " File logging disabled due to error."
    root_logger.info(initial_message)

    # --- Optional: Silence noisy libraries ---
    # Example: Reduce verbosity of specific libraries if needed
    # logging.getLogger("urllib3").setLevel(logging.WARNING)
    # logging.getLogger("chromadb").setLevel(logging.WARNING)


# --- How to use in other modules ---
# import logging
# logger = logging.getLogger(__name__) # Use module name for logger
#
# logger.debug("This is a debug message.")
# logger.info("This is an info message.")
# logger.warning("This is a warning message.")
# logger.error("This is an error message.")
# logger.critical("This is a critical message.")

# --- Example of calling setup (usually done in main.py) ---
# if __name__ == "__main__":
#     print("Setting up logging directly...")
#     setup_logging()
#     logging.info("Logging setup complete from direct call.")
#     logging.warning("This is a test warning.")
#     # In other files, just import logging and get logger:
#     logger = logging.getLogger("my_module_test")
#     logger.info("Info message from 'my_module_test'")

