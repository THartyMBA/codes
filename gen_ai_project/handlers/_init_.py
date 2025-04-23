# handlers/__init__.py

# This file makes the 'handlers' directory a Python package.

# Expose the main handler classes for easier importing from the package level
from .chat_handler import ChatHandler
from .events_handler import EventHandler
from .scheduler_handler import SchedulerHandler
from .goal_handler import GoalHandler
from .monitoring_handler import MonitoringHandler
from .pipeline_handler import PipelineHandler
from .admin_handler import AdminHandler
from .knowledge_handler import KnowledgeHandler
from .file_handler import FileHandler

# This allows imports like:
# from handlers import ChatHandler
# instead of:
# from handlers.chat_handler import ChatHandler
