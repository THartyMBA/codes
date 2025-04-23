# Modular GenAI Agent System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, extensible, and modular foundation for building AI agent systems using Python, LangChain, FastAPI, and Streamlit. Designed to be easily configurable and extendable for various use cases, allowing developers to plug in components and get started quickly.

## Overview

This project provides a structured framework for creating sophisticated AI agents capable of interacting with databases, performing data analysis and visualization, generating reports, managing files, executing complex goals, running predefined pipelines, monitoring systems, and maintaining conversational context.

The core architecture emphasizes separation of concerns through distinct **Handlers** responsible for different modes of operation (chat, events, scheduling, goals, etc.) and specialized **Tool Agents** that provide specific capabilities. These components interact via a central **CoreAgentServices** module, which manages shared resources like the LLM, database connections, vector stores, and memory.

## Key Features

*   **Modular Handler Architecture:**
    *   `ChatHandler`: Manages interactive user conversations.
    *   `EventHandler`: Processes events from internal or external sources (e.g., file changes, API triggers, monitor alerts).
    *   `SchedulerHandler`: Executes tasks based on time schedules (interval, cron) using APScheduler.
    *   `GoalHandler`: Manages long-running, stateful tasks using LangGraph.
    *   `PipelineHandler`: Executes predefined sequences of steps (workflows).
    *   `MonitoringHandler`: Proactively checks system/data conditions and triggers actions.
    *   `KnowledgeHandler`: Manages interaction with the vector store (RAG).
    *   `AdminHandler`: Provides administrative functions (health checks, config, etc.).
*   **Core Services Abstraction:** Centralized management of shared resources:
    *   LLM Integration (Defaults to Ollama via `langchain-community`).
    *   Database Connection (SQLAlchemy, supporting SQLite, PostgreSQL, MySQL).
    *   Vector Store (ChromaDB with SentenceTransformer embeddings).
    *   Conversational Memory (Mem0).
*   **Specialized Tool Agents:** Pre-built agents for common tasks:
    *   `SQLAgent`: Natural language to SQL querying.
    *   `VisualizationAgent`: Generates plots (Matplotlib/Seaborn).
    *   `ReportingAgent`: Creates PDF reports from HTML (WeasyPrint).
    *   `ModelingAgent`: Basic ML model training/evaluation (Scikit-learn).
    *   `ForecastingAgent`: Time series forecasting (Statsmodels SARIMAX).
    *   File Management Toolkit (via LangChain).
*   **RAG (Retrieval-Augmented Generation):** `KnowledgeHandler` manages adding documents and retrieving context from ChromaDB.
*   **Configurable Workflows:**
    *   Define multi-step pipelines in `config/pipelines.yaml`.
    *   Define monitoring checks, rules, and actions in `config/monitors.yaml`.
*   **Goal Execution:** Define complex, stateful goals using LangGraph (example: `graphs/website_monitor.py`).
*   **API & UI:**
    *   **FastAPI Backend:** Exposes system functionality via a REST API (includes interactive docs).
    *   **Streamlit Frontend:** Provides a web interface for chat, goal/pipeline management, and admin tasks.
*   **Externalized Configuration:** Uses `.env` for secrets/environment settings and YAML files for pipelines/monitors.
*   **Asynchronous Design:** Leverages `asyncio` for concurrent handling of requests, events, and background tasks.

## Architecture Overview

+---------------------+ +-----------------+ +-----------------+ | Streamlit UI |----->| FastAPI API |<-----| External Systems| +---------------------+ +-------+---------+ +-----------------+ | | (Requests, Events) v +------------------------------------+-----------------------------------+ | Handlers | | +----------+ +---------+ +-----------+ +--------+ +----------+ | | | Chat | | Event | | Scheduler | | Goal | | Pipeline | ...etc | | +----+-----+ +----+----+ +-----+-----+ +---+----+ +-----+----+ | +------|------------|------------|-----------|------------|---------------+ | | | | | v v v v v +------------------------------------------------------------------------+ | CoreAgentServices | | +---------+ +--------+ +---------+ +--------+ +----------+ +---------+ | | | LLM | | DB Cnx | | Vector | | Memory | | Executor | | Handler | | | |(Ollama) | |(SQLAlch)| | (Chroma)| | (Mem0) | |(LangChain)| | Refs | | | +---------+ +--------+ +---------+ +--------+ +----------+ +---------+ | | | | | | +---------------------------------------------+-------------------+ | | | +---------------------------------------------+ | | Tool Agents / Toolkit | | | +-----+ +-----+ +------+ +-------+ +------+ | | | | SQL | | Viz | | Rept | | Model | | File | | ...etc | | +-----+ +-----+ +------+ +-------+ +------+ | +------------------------------------------------------------------------+


*   **UI/API:** Entry points for users and external systems.
*   **Handlers:** Orchestrate workflows based on triggers (user input, events, schedules). They decide *what* needs to be done.
*   **CoreAgentServices:** Provides access to shared resources (LLM, DB, Memory, Vector Store) and initialized instances of handlers/agents. Holds the core LangChain `AgentExecutor`.
*   **Tool Agents/Toolkit:** Perform specific, well-defined tasks using the shared resources.

## Getting Started

### Prerequisites

*   **Python:** 3.10 or higher recommended.
*   **Git:** To clone the repository.
*   **Ollama:** (Recommended LLM Backend) Ensure Ollama is installed and running. Pull your desired model (e.g., `ollama pull mistral`). See Ollama GitHub.
*   **OS Dependencies for WeasyPrint:** The `ReportingAgent` uses WeasyPrint, which requires system libraries like Pango, Cairo, and GDK-PixBuf. Follow the installation guide for your OS: WeasyPrint Installation.

### Setup Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/<your-username>/<your-repo-name>.git # Replace with your repo URL
    cd <your-repo-name>
    ```

2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This may take some time, especially for libraries like PyTorch if pulled by SentenceTransformers).*

4.  **Configure Environment (`.env`):**
    *   Copy the template: `cp .env.example .env` (or manually create `.env`).
    *   Edit the `.env` file:
        *   **Crucially, set `ADMIN_API_KEY`**. Generate a strong secret key (e.g., `python -c "import secrets; print(secrets.token_hex(32))"`). Admin features require this key.
        *   Review `OLLAMA_MODEL` and ensure it matches a model you have pulled in Ollama.
        *   Configure database settings (`DB_TYPE`, `DB_NAME`, or `DATABASE_URL`) if not using the default SQLite (`sample_company.db`).
        *   Set other API keys (e.g., `OPENAI_API_KEY`) if needed for specific features.
        *   Adjust paths (`WORKSPACE_DIR`, `LOG_DIRECTORY`) if desired (defaults are usually fine).

5.  **Prepare Sample Data (Optional but Recommended):**
    *   Ensure the default SQLite database `sample_company.db` exists in the root (or configure your DB).
    *   Place sample files like `employees.csv`, `company_policy.txt`, `sample_ts_data.csv` into the `workspace/` directory (or the configured `WORKSPACE_DIR`) if you want to use the default examples in pipelines, monitors, or for initial knowledge loading.

## Configuration

*   **`.env`:** Stores environment-specific settings, secrets, and API keys. See `.env.example` for details.
*   **`config/pipelines.yaml`:** Defines named sequences of steps (pipelines) that can be executed. Each step typically involves calling a tool/method from `CoreAgentServices`. Uses `{{context.key}}` and `{{initial_context.key}}` for passing data between steps.
*   **`config/monitors.yaml`:** Defines proactive monitoring checks. Each monitor specifies a `check` (e.g., SQL query, tool call), a `rule` (e.g., threshold, equality) to evaluate the check result, and an `action` (e.g., enqueue event, direct tool call) to perform if the rule is met.
*   **`utils/config.py`:** Loads settings from `.env` and provides default values.

## Running the Application

Execute the main run script:

```bash
python run.py

This will typically:

Start the FastAPI backend server (usually on http://127.0.0.1:8000).
API documentation (Swagger UI) is available at http://127.0.0.1:8000/docs.
OpenAPI schema at http://127.0.0.1:8000/openapi.json.
Start the Streamlit web UI (usually on http://localhost:8501).
Check the console output for the exact URLs.

Usage
Streamlit UI (http://localhost:8501)
Chat Tab: Interact directly with the agent. Ask questions, request data analysis, report generation, etc.
Goals Tab: Define and start long-running goals (e.g., website monitoring). View the status of active goals and cancel them.
Pipelines Tab: Select predefined pipelines, provide initial context (if needed), and trigger runs. View the status and results of pipeline runs.
Admin Tab: (Requires Admin API Key set in sidebar) Perform administrative actions like checking system health, triggering knowledge re-indexing, and clearing user memory (experimental).
FastAPI (http://127.0.0.1:8000)
Use the /docs endpoint for interactive API testing.
Integrate with the API endpoints programmatically:
/chat: Send user messages.
/events: Enqueue events from external systems.
/goals: Start, get status, cancel goals.
/pipelines: Start, get status, cancel pipeline runs.
/admin: Access administrative functions (requires X-API-Key header).
/status: Basic health check.
Extending the System
This framework is designed for extension:

Adding a New Tool Agent:
Create a new Python file in the agents/ directory (e.g., my_tool_agent.py).
Define a class (optionally inheriting from BaseAgent).
Implement the core logic (e.g., interacting with an external API).
In core/core_services.py:
Import your new agent class.
Initialize it within the __init__ method.
Create an async wrapper method (e.g., _run_my_tool_async) if the agent's core method is synchronous.
Add a langchain_core.tools.tool definition in the _get_tools method, pointing to your wrapper.
Adding a New Handler:
Create a new Python file in the handlers/ directory (e.g., my_handler.py).
Define the handler class, taking CoreAgentServices in its __init__.
Implement the handler's logic (e.g., processing specific events, managing a new type of background task).
Import and initialize the handler in api.py during the startup_event.
Add logic to stop the handler in the shutdown_event if necessary.
Add API endpoints in api.py to interact with the new handler if needed.
Adding a Pipeline:
Edit config/pipelines.yaml.
Add a new top-level key (the pipeline ID).
Define a list of steps under the ID.
Each step needs at least a tool key referencing a method available via CoreAgentServices (e.g., _run_sql_query_tool_async, add_memory, knowledge_handler.add_source).
Use params to specify arguments, using {{initial_context.key}} or {{context.previous_output_key}} for dynamic values.
Use output_key to name the result of a step for use in subsequent steps.
Adding a Monitor:
Edit config/monitors.yaml.
Add a new top-level key (the monitor ID).
Define check, rule, and action dictionaries according to the desired logic. Reference available check types (sql, db_ping, tool_call) and action types (enqueue_event, direct_call).
Adding a Goal:
Create a new Python file in the graphs/ directory (e.g., my_goal_graph.py).
Define the TypedDict state for your goal.
Implement node functions (async recommended).
Define conditional edge logic functions.
Create a builder function (e.g., get_my_goal_app(checkpointer)) that constructs and compiles the StateGraph.
In handlers/goal_handler.py:
Import your builder function.
Update _get_graph_app to call your builder function based on a new goal_type.
Update _parse_goal_to_state to handle parsing the initial state for your new goal_type.
Testing
Unit and integration tests are recommended for ensuring robustness and facilitating contributions. Contributions in this area are welcome!

Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests. For major changes, please open an issue first to discuss the proposed changes.



License
This project is licensed under the MIT License - see the LICENSE file for details.