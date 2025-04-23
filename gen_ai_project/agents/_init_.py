# This file makes the 'agents' directory a Python package.

# You can optionally expose key classes or functions directly at the package level
# for easier importing elsewhere.
from .supervisor_agent import SupervisorAgent
from .sql_agent import SQLAgent
from .visualization_agent import VisualizationAgent
from .modeling_agent import ModelingAgent
from .forecasting_agent import ForecastingAgent

# You could add other agents here if desired, e.g.:
# from .email_agent import EmailAgent

# This allows imports like: from agents import SupervisorAgent