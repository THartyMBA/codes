# main.py - Entry point for the Gen AI Project

import os
import time
import schedule
from datetime import datetime
import sqlite3 # For dummy DB setup
import pandas as pd # For dummy TS data
import numpy as np # For dummy TS data
import sys

# --- Project Imports ---
# Ensure the project root is implicitly in the Python path when running main.py from the root
try:
    from agents.supervisor_agent import SupervisorAgent
    from utils import config # Import config module
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure you are running this script from the 'gen_ai_project' root directory,")
    print("and that the 'agents' and 'utils' directories with '__init__.py' files exist.")
    sys.exit(1)

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Gen AI Supervisor Application ---")

    # --- Database Setup (SQLite Example - Uses config) ---
    if config.DB_URI and config.DB_TYPE == "sqlite" and not os.path.exists(config.DB_NAME):
        print(f"Creating dummy SQLite database: {config.DB_NAME}")
        try:
            conn = sqlite3.connect(config.DB_NAME)
            cursor = conn.cursor()
            # Simple example schema (adjust if your SQLAgent expects something different)
            cursor.execute('''CREATE TABLE IF NOT EXISTS employees (
                                id INTEGER PRIMARY KEY,
                                name TEXT,
                                department TEXT,
                                salary REAL,
                                experience_years INTEGER
                             )''')
            cursor.executemany("INSERT INTO employees (name, department, salary, experience_years) VALUES (?, ?, ?, ?)",
                               [('Alice','Eng',90000,5), ('Bob','Sales',75000,3), ('Charlie','Eng',95000,7),
                                ('David','Sales',80000,4), ('Eve','HR',65000,2), ('Frank','Eng',110000,10),
                                ('Grace','HR',70000,3)])
            conn.commit()
            conn.close()
            print("Dummy database created and populated.")
        except Exception as e:
            print(f"Error creating dummy database: {e}")
    elif config.DB_URI:
        print(f"Using existing database: {config.DB_NAME}")
    else:
        print("No database configured (DB_URI is not set in config). SQL Agent will be unavailable.")

    # --- Initialize Supervisor ---
    print("\n--- Initializing Supervisor Agent ---")
    try:
        # Pass configuration values during initialization
        supervisor = SupervisorAgent(
            model_name=config.OLLAMA_MODEL,
            temperature=0.1, # Example temperature
            db_uri=config.DB_URI
        )
        # Workspace directory is now managed within SupervisorAgent, using an absolute path
        print(f"Supervisor initialized. Workspace: {supervisor.workspace_dir}")
    except Exception as e:
        print(f"FATAL: Failed to initialize SupervisorAgent: {e}")
        sys.exit(1)


    # --- Create Dummy Knowledge File (in workspace) ---
    KNOWLEDGE_FILE_NAME = "company_policy.txt" # Just the filename
    knowledge_file_path = os.path.join(supervisor.workspace_dir, KNOWLEDGE_FILE_NAME)
    if not os.path.exists(knowledge_file_path):
        print(f"\n--- Creating Dummy Knowledge File: {KNOWLEDGE_FILE_NAME} in workspace ---")
        policy_content = """
Company Policy Document - v1.2

Remote Work:
Employees in Engineering and Marketing departments are eligible for full remote work.
Sales department requires hybrid work (3 days in office).
HR department requires full-time in-office presence.
All remote employees must maintain core working hours of 10 AM to 4 PM local time.

Vacation Policy:
Standard vacation allowance is 20 days per year, accrued monthly.
Unused vacation days up to 5 can be carried over to the next year.
Requests must be submitted via the HR portal at least 2 weeks in advance.

Project Phoenix:
This is a high-priority project for Q3/Q4.
Lead Engineer: Charlie. Project Manager: David.
Weekly status reports are due every Friday by 5 PM EST.
Key deliverable: Beta release by November 15th.
"""
        try:
            with open(knowledge_file_path, "w") as f:
                f.write(policy_content)
            print(f"Knowledge file created at {knowledge_file_path}")
        except Exception as e:
            print(f"Error creating knowledge file: {e}")

    # --- Add Knowledge using the Tool (Run once check) ---
    if os.path.exists(knowledge_file_path):
        # Check if knowledge might already be added (simple check based on Mem0)
        try:
            mem_search = supervisor.mem0_memory.search(query=f"Ingested.*{KNOWLEDGE_FILE_NAME}", user_id="supervisor_system", limit=1)
            if not mem_search:
                print("\n--- Task: Add Knowledge to KB (First time setup) ---")
                # Pass the filename relative to the workspace to the tool
                task_add_knowledge = f"Please add the document '{KNOWLEDGE_FILE_NAME}' to the knowledge base."
                result_add_knowledge = supervisor.run(task_add_knowledge, user_id="admin_user")
                print("\nAdd Knowledge Task Result:", result_add_knowledge.get('final_output', result_add_knowledge))
            else:
                print(f"\n--- Knowledge file '{KNOWLEDGE_FILE_NAME}' likely already ingested (found in memory). Skipping addition. ---")
        except Exception as e:
             print(f"Error during knowledge ingestion check/run: {e}")
    else:
        print(f"\nSkipping knowledge addition task ({KNOWLEDGE_FILE_NAME} not found in workspace).")


    # --- CHOOSE EXECUTION MODE ---
    # Set this flag to True to run the autonomous scheduler, False for interactive examples.
    RUN_AUTONOMOUS_SCHEDULER = False

    if RUN_AUTONOMOUS_SCHEDULER:
        # --- Autonomous Scheduler Mode ---
        print("\n--- Starting Autonomous Scheduler Mode ---")
        print("NOTE: This mode will run scheduled tasks indefinitely.")
        print("      Interactive chat examples below will be skipped.")
        print("      Press Ctrl+C to stop the scheduler.")

        # Define Scheduled Tasks (functions calling the supervisor's autonomous method)
        def run_daily_summary_task():
            print(f"\n[{datetime.now()}] Triggering Daily Summary Task...")
            task_prompt = "Query the database for the total number of employees in each department and list them."
            supervisor._execute_autonomous_task(task_prompt, task_id="daily_dept_summary")

        def run_knowledge_check_task():
             print(f"\n[{datetime.now()}] Triggering Knowledge Check Task...")
             task_prompt = "What is the company policy on remote work for the Sales department?"
             supervisor._execute_autonomous_task(task_prompt, task_id="kb_remote_work_check")

        # Schedule the Tasks
        # Use more realistic schedules for production
        schedule.every(2).minutes.do(run_daily_summary_task) # Example: Run summary every 2 minutes for testing
        schedule.every(3).minutes.do(run_knowledge_check_task) # Example: Run KB check every 3 minutes for testing

        print("\n--- Starting Scheduler ---")
        print(f"Scheduled tasks: {schedule.get_jobs()}")

        # Run the Scheduler Loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(1) # Check every second
        except KeyboardInterrupt:
            print("\nScheduler stopped by user.")
        except Exception as e:
             print(f"\nAn error occurred in the scheduler loop: {e}")

    else:
        # --- Interactive Chat Mode ---
        print("\n--- Starting Interactive Chat Mode ---")
        print("NOTE: Autonomous scheduler is disabled. Running example interactions.")

        # Example 1: Ask question requiring Knowledge Base (RAG)
        print("\n--- Task: Ask question requiring KB ---")
        task_rag1 = "What is the vacation policy regarding carry-over days?"
        result_rag1 = supervisor.run(task_rag1, user_id="employee_alice")
        print("\nKB Question 1 Result:", result_rag1.get('final_output', 'No final output found.'))

        # Example 2: Ask another KB question
        print("\n--- Task: Ask another question requiring KB ---")
        task_rag2 = "Who is the project manager for Project Phoenix?"
        result_rag2 = supervisor.run(task_rag2, user_id="employee_bob")
        print("\nKB Question 2 Result:", result_rag2.get('final_output', 'No final output found.'))

        # Example 3: SQL Query (if DB is configured)
        if config.DB_URI:
            print("\n--- Task: SQL Query ---")
            task_sql = "Who are the employees in the Engineering department earning over 92000?"
            result_sql = supervisor.run(task_sql, user_id="hr_manager")
            print("\nSQL Task Result:", result_sql.get('final_output', 'No final output found.'))
        else:
            print("\nSkipping SQL Query task (No DB_URI configured).")

        # Example 4: Forecasting (Create dummy data if needed)
        TIME_SERIES_DATA_FILENAME = "sample_ts_data.csv" # Just the filename
        ts_file_path = os.path.join(supervisor.workspace_dir, TIME_SERIES_DATA_FILENAME)

        if not os.path.exists(ts_file_path):
            print(f"\n--- Creating dummy time series data: {TIME_SERIES_DATA_FILENAME} in workspace ---")
            try:
                dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
                volume = (np.sin(np.arange(90) * 2 * np.pi / 7) * 10 +
                          np.linspace(50, 80, 90) +
                          np.random.normal(0, 5, 90))
                ts_df = pd.DataFrame({'Date': dates, 'CallVolume': volume.astype(int)})
                ts_df.to_csv(ts_file_path, index=False)
                print(f"Dummy file '{TIME_SERIES_DATA_FILENAME}' created.")
            except Exception as e:
                print(f"Error creating dummy TS data: {e}")

        if os.path.exists(ts_file_path):
            print("\n--- Task: Time Series Forecasting ---")
            # Pass the filename relative to the workspace to the tool via the prompt
            task_forecast = (f"Generate a 7-day forecast for 'CallVolume' using data from '{TIME_SERIES_DATA_FILENAME}'. "
                             f"The time column is 'Date' and the frequency is daily ('D'). Save the forecast data and model.")
            result_forecast = supervisor.run(task_forecast, user_id="ops_manager")
            print("\nForecasting Task Result:", result_forecast.get('final_output', 'No final output found.'))
        else:
             print("\nSkipping Forecasting task (data file not found/created).")

        # Example 5: Visualization (using the dummy TS data)
        if os.path.exists(ts_file_path):
            print("\n--- Task: Create Visualization ---")
            viz_output_filename = "call_volume_plot.png"
            # Pass filenames relative to workspace
            task_viz = (f"Create a line plot of 'CallVolume' over time using data from '{TIME_SERIES_DATA_FILENAME}'. "
                        f"Save the plot as '{viz_output_filename}'.")
            result_viz = supervisor.run(task_viz, user_id="analyst_grace")
            print("\nVisualization Task Result:", result_viz.get('final_output', 'No final output found.'))
            # Check if file exists
            viz_full_path = os.path.join(supervisor.workspace_dir, viz_output_filename)
            if os.path.exists(viz_full_path):
                 print(f"Visualization file created at: {viz_full_path}")
            else:
                 print(f"Visualization file NOT found at: {viz_full_path}")
        else:
             print("\nSkipping Visualization task (data file not found/created).")

        # Example 6: File Listing (using FileManagement tool)
        print("\n--- Task: List files in workspace ---")
        task_files = "List all files currently in my workspace directory."
        result_files = supervisor.run(task_files, user_id="admin_user")
        print("\nFile Listing Task Result:", result_files.get('final_output', 'No final output found.'))


    print("\n--- Application Execution Finished ---")

