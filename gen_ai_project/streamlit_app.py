# streamlit_app.py (Place in the project root: gen_ai_project/)

import streamlit as st
import requests # To make API calls
import uuid
import json
import pandas as pd
from datetime import datetime

# --- Configuration ---
# API endpoint (should match where your FastAPI app is running)
API_BASE_URL = "http://127.0.0.1:8000" # Default for local run
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
STATUS_ENDPOINT = f"{API_BASE_URL}/status"

# --- Page Config ---
st.set_page_config(page_title="Supervisor Agent Chat", layout="wide")
st.title("🤖 Supervisor Agent Interface")

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [] # Store chat history {role: "user"/"assistant", content: "..."}
    st.session_state.user_id = f"streamlit_{st.session_state.session_id[:8]}" # Unique ID per session

st.sidebar.markdown(f"**Session ID:** `{st.session_state.session_id}`")
st.sidebar.markdown(f"**User ID:** `{st.session_state.user_id}`")

# --- API Communication Functions ---
def send_chat_message(user_input: str, user_id: str):
    """Sends message to the backend API and gets response."""
    payload = {"user_input": user_input, "user_id": user_id}
    try:
        response = requests.post(CHAT_ENDPOINT, json=payload, timeout=120) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json() # Expects JSON like {"final_output": "...", "error": null}
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {e}")
        return {"final_output": "Error: Could not connect to the agent API.", "error": str(e)}
    except json.JSONDecodeError:
        st.error("API Error: Invalid response format received.")
        return {"final_output": "Error: Received invalid response from API.", "error": "JSONDecodeError"}

def get_agent_status():
    """Fetches status from the backend API."""
    try:
        response = requests.get(STATUS_ENDPOINT, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error fetching status: {e}")
        return None
    except json.JSONDecodeError:
        st.error("API Error: Invalid status response format.")
        return None

# --- UI Layout ---
col1, col2 = st.columns([3, 1]) # Chat area takes more space

# --- Column 1: Chat Interface ---
with col1:
    st.subheader("Chat with Supervisor")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask the supervisor..."):
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send to backend and get response
        with st.spinner("Supervisor is thinking..."):
            api_response = send_chat_message(prompt, st.session_state.user_id)

        # Display assistant response
        response_content = api_response.get("final_output", "Sorry, I encountered an issue.")
        error_content = api_response.get("error")

        with st.chat_message("assistant"):
            st.markdown(response_content)
            if error_content:
                st.error(f"Agent reported an error: {error_content}")

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response_content})

# --- Column 2: Status Area ---
with col2:
    st.subheader("Agent Status")
    if st.button("Refresh Status"):
        status_data = get_agent_status()
        if status_data:
            st.session_state.last_status = status_data # Store status in session state
        else:
            # Clear previous status if refresh failed
            if "last_status" in st.session_state:
                del st.session_state.last_status

    # Display last fetched status
    if "last_status" in st.session_state:
        status = st.session_state.last_status
        st.markdown(f"**Scheduler Running:** {'✅ Yes' if status.get('scheduler_running') else '❌ No'}")
        st.markdown(f"**Event Queue Size:** {status.get('event_queue_size', 'N/A')}")

        st.markdown("**Scheduled Jobs:**")
        jobs = status.get("scheduled_jobs", [])
        if jobs:
            # Format job data for display
            jobs_df_data = []
            for job in jobs:
                 try:
                     # Attempt to parse next run time for better display
                     next_run = datetime.fromisoformat(job.get("next_run_time", "")).strftime('%Y-%m-%d %H:%M:%S %Z') if job.get("next_run_time") else "N/A"
                 except:
                     next_run = job.get("next_run_time", "N/A") # Fallback to string if parsing fails
                 jobs_df_data.append({
                     "ID": job.get("id"),
                     "Next Run": next_run,
                     "Trigger": job.get("trigger")
                 })
            st.dataframe(pd.DataFrame(jobs_df_data), use_container_width=True)
        else:
            st.caption("No scheduled jobs found.")

        st.markdown("**Running Goals:**")
        goals = status.get("running_goals", [])
        if goals:
            for goal_id in goals:
                st.caption(f"- `{goal_id}`")
        else:
            st.caption("No active goals found.")

    else:
        st.caption("Click 'Refresh Status' to load.")

