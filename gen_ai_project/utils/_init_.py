# This file makes the 'utils' directory a Python package.

# Expose the config module and specific utility functions
from . import config
from .document_processing import load_and_split_documents

# You could also expose specific config variables if preferred, e.g.:
# from .config import OLLAMA_MODEL, DB_URI, WORKSPACE_DIR

# This allows imports like:
# import utils.config
# from utils import load_and_split_documents
# from utils import config (if using the line above)
# from utils import OLLAMA_MODEL (if exposing specific variables)