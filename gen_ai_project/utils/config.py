import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists in the project root.
# Create a file named '.env' in your project root (gen_ai_project/) for sensitive info.
# Example .env content:
# OLLAMA_MODEL="mistral"
# DB_TYPE="sqlite"
# DB_NAME="my_company.db"
# OPENAI_API_KEY="sk-..."
# CHROMA_DB_PATH="./vector_store" # Override default path

# Determine the project root directory (assuming config.py is in utils/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- LLM Configuration ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral") # Default to "mistral" if not set

# --- Database Configuration ---
# Set DB_TYPE to "sqlite", "postgresql", "mysql", etc. in .env or here
DB_TYPE = os.getenv("DB_TYPE", "sqlite")
# For SQLite, DB_NAME is the filename (relative to project root). For others, it's the database name.
DB_NAME = os.getenv("DB_NAME", "sample_company.db")
# For non-SQLite databases, set these in .env
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432") # Example default for PostgreSQL
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

# Construct the Database URI (SQLAlchemy connection string)
DB_URI = None
if DB_TYPE == "sqlite":
    # Construct absolute path for SQLite DB relative to project root
    sqlite_path = os.path.join(PROJECT_ROOT, DB_NAME)
    DB_URI = f"sqlite:///{sqlite_path}"
elif DB_TYPE == "postgresql":
    # Ensure you have 'psycopg2-binary' installed
    DB_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
elif DB_TYPE == "mysql":
    # Ensure you have 'mysql-connector-python' installed
    DB_URI = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# Add other database types (e.g., SQL Server with pyodbc) as needed
elif os.getenv("DATABASE_URL"): # Allow setting the full URL directly via env var
    DB_URI = os.getenv("DATABASE_URL")

# --- RAG/Vector Store Configuration ---
# Default path for ChromaDB persistent storage (relative to project root)
_default_chroma_path = os.path.join(PROJECT_ROOT, "chroma_db_supervisor")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", _default_chroma_path)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "supervisor_knowledge_base")

# Embedding model (default to SentenceTransformer, free and local)
# Options: "all-MiniLM-L6-v2", "all-mpnet-base-v2", or OpenAI models like "text-embedding-ada-002"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --- Agent Workspace Configuration ---
# Default directory for agents to read/write files (relative to project root)
_default_workspace_path = os.path.join(PROJECT_ROOT, "workspace")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", _default_workspace_path)

# --- API Keys (Loaded from .env - DO NOT HARDCODE KEYS HERE) ---
# These are examples; the actual keys should be in your .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
# Add other API keys as needed

# --- Other Configurations ---
# Default temperature for LLM generation
DEFAULT_LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", 0.1))

# Number of RAG results to retrieve
RAG_RESULTS_COUNT = int(os.getenv("RAG_RESULTS_COUNT", 3))

# Number of Mem0 results to retrieve
MEM0_RESULTS_COUNT = int(os.getenv("MEM0_RESULTS_COUNT", 5))


# --- You can add validation or print statements here for debugging ---
# print(f"--- Configuration Loaded ---")
# print(f"Project Root: {PROJECT_ROOT}")
# print(f"OLLAMA_MODEL: {OLLAMA_MODEL}")
# print(f"DB_TYPE: {DB_TYPE}")
# print(f"DB_NAME: {DB_NAME}")
# # print(f"DB_URI: {DB_URI}") # Be careful printing URIs with passwords
# print(f"CHROMA_DB_PATH: {CHROMA_DB_PATH}")
# print(f"COLLECTION_NAME: {COLLECTION_NAME}")
# print(f"EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
# print(f"WORKSPACE_DIR: {WORKSPACE_DIR}")
# print(f"OpenAI Key Loaded: {'Yes' if OPENAI_API_KEY else 'No'}")
# print(f"--------------------------")

# Ensure workspace and chroma directories exist
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)