# handlers/__init__.py

# This file makes the 'handlers' directory a Python package.

# Expose the main handler classes for easier importing from the package level
from .chat_handler import ChatHandler
from .events_handler import EventHandler
from .scheduler_handler import SchedulerHandler

# This allows imports like:
# from handlers import ChatHandler
# instead of:
# from handlers.chat_handler import ChatHandler
