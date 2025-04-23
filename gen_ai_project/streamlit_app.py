# streamlit_app.py (Place in the project root: gen_ai_project/)

import streamlit as st
import requests # To make API calls
import uuid
import json
import pandas as pd
from datetime import datetime
import time

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000" # Default for local run
# API Endpoints
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
EVENTS_ENDPOINT = f"{API_BASE_URL}/events" # Renamed from /enqueue
GOALS_ENDPOINT = f"{API_BASE_URL}/goals"
PIPELINES_ENDPOINT = f"{API_BASE_URL}/pipelines"
ADMIN_OVERVIEW_ENDPOINT = f"{API_BASE_URL}/admin/overview"
ADMIN_HEALTH_ENDPOINT = f"{API_BASE_URL}/admin/health"
ADMIN_REINDEX_ENDPOINT = f"{API_BASE_URL}/admin/knowledge/reindex"
ADMIN_CLEAR_MEMORY_ENDPOINT = f"{API_BASE_URL}/admin/memory"

# --- Page Config ---
st.set_page_config(page_title="Agent Interface", layout="wide")
st.title("🤖 GenAI Agent System Interface")

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [] # Store chat history {role: "user"/"assistant", content: "..."}
    st.session_state.user_id = f"streamlit_{st.session_state.session_id[:8]}" # Unique ID per session
    st.session_state.admin_api_key = "" # Store API key if needed for admin actions
    st.session_state.active_goals = {} # Store goal status {goal_id: status_dict}
    st.session_state.active_pipelines = {} # Store pipeline status {run_id: status_dict}

# --- Sidebar ---
st.sidebar.markdown(f"**Session ID:** `{st.session_state.session_id}`")
st.sidebar.markdown(f"**User ID:** `{st.session_state.user_id}`")
st.sidebar.divider()
# Placeholder for Admin API Key - Implement proper auth for production
st.sidebar.subheader("Admin Access (Demo)")
st.session_state.admin_api_key = st.sidebar.text_input("Admin API Key", type="password", value=st.session_state.admin_api_key)
st.sidebar.warning("Admin endpoints require a valid API key configured on the backend.")
st.sidebar.divider()
st.sidebar.subheader("System Overview")
if st.sidebar.button("🔄 Refresh Overview"):
    # Fetch overview using the admin endpoint (requires key)
    headers = {"X-API-Key": st.session_state.admin_api_key} if st.session_state.admin_api_key else {}
    try:
        response = requests.get(ADMIN_OVERVIEW_ENDPOINT, headers=headers, timeout=10)
        if response.status_code == 403:
             st.sidebar.error("Access Denied. Invalid/Missing Admin API Key?")
             st.session_state.last_overview = {"error": "Access Denied"}
        elif response.status_code == 503:
             st.sidebar.error("Admin Handler not available on backend.")
             st.session_state.last_overview = {"error": "Admin Handler Unavailable"}
        else:
             response.raise_for_status()
             st.session_state.last_overview = response.json().get("details", {})
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"API Error fetching overview: {e}")
        if "last_overview" in st.session_state: del st.session_state.last_overview
    except Exception as e:
         st.sidebar.error(f"Error processing overview: {e}")
         if "last_overview" in st.session_state: del st.session_state.last_overview

# Display last fetched overview
if "last_overview" in st.session_state:
    overview = st.session_state.last_overview
    if overview.get("error"):
         st.sidebar.caption(f"Failed to load overview: {overview['error']}")
    else:
        st.sidebar.caption(f"**LLM:** `{overview.get('llm_model', 'N/A')}`")
        st.sidebar.caption(f"**Workspace:** `{overview.get('workspace_path', 'N/A')}`")
        handlers_status = overview.get("handlers_status", {})
        if hs := handlers_status.get("scheduler"): st.sidebar.caption(f"**Scheduler:** {'Running' if hs.get('running') else 'Stopped'} ({hs.get('job_count', 0)} jobs)")
        if hs := handlers_status.get("event_processor"): st.sidebar.caption(f"**Events:** {'Running' if hs.get('running') else 'Stopped'} (Queue: {hs.get('queue_size', 0)})")
        if hs := handlers_status.get("goal_handler"): st.sidebar.caption(f"**Goals:** {hs.get('active_goals', 0)} active")
        if hs := handlers_status.get("pipeline_handler"): st.sidebar.caption(f"**Pipelines:** {hs.get('active_pipelines', 0)} active")
else:
    st.sidebar.caption("Click 'Refresh Overview' to load.")


# --- API Communication Functions ---
# (Includes error handling and optional API key header)

def _make_request(method, url, headers=None, **kwargs):
    """Helper function for making API requests with error handling."""
    if headers is None: headers = {}
    # Add API key if provided and needed (e.g., for admin endpoints)
    if st.session_state.admin_api_key and "/admin/" in url:
         headers["X-API-Key"] = st.session_state.admin_api_key

    try:
        response = requests.request(method, url, headers=headers, **kwargs)
        if response.status_code == 403:
             st.error(f"API Access Denied (403) for {url}. Check Admin API Key?")
             return None # Indicate auth failure
        response.raise_for_status() # Raise HTTPError for other bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error ({method} {url}): {e}")
        return None
    except json.JSONDecodeError:
        st.error(f"API Error ({method} {url}): Invalid JSON response.")
        return None
    except Exception as e:
         st.error(f"Unexpected Error ({method} {url}): {e}")
         return None

# --- Specific API Call Functions ---

def send_chat_message(user_input: str, user_id: str):
    payload = {"user_input": user_input, "user_id": user_id}
    response_data = _make_request("post", CHAT_ENDPOINT, json=payload, timeout=180) # Longer timeout for chat
    return response_data or {"final_output": "Error: API request failed.", "error": "Request failed"}

def enqueue_event(event_type: str, event_data: dict):
    payload = {"event_type": event_type, "event_data": event_data}
    return _make_request("post", EVENTS_ENDPOINT, json=payload, timeout=10)

def start_goal(goal_type: str, description: str):
    payload = {"goal_type": goal_type, "goal_description": description}
    return _make_request("post", GOALS_ENDPOINT, json=payload, timeout=20)

def get_goal_status(goal_id: str):
    return _make_request("get", f"{GOALS_ENDPOINT}/{goal_id}/status", timeout=10)

def cancel_goal(goal_id: str):
    return _make_request("delete", f"{GOALS_ENDPOINT}/{goal_id}", timeout=15)

def start_pipeline(pipeline_id: str, initial_context: Optional[dict]):
    payload = {"pipeline_id": pipeline_id, "initial_context": initial_context or {}}
    return _make_request("post", f"{PIPELINES_ENDPOINT}/{pipeline_id}/run", json=payload, timeout=20)

def get_pipeline_status(run_id: str):
    return _make_request("get", f"{PIPELINES_ENDPOINT}/runs/{run_id}", timeout=10)

def cancel_pipeline(run_id: str):
    return _make_request("delete", f"{PIPELINES_ENDPOINT}/runs/{run_id}", timeout=15)

def get_admin_overview_data(): # Separate function for admin overview call
    return _make_request("get", ADMIN_OVERVIEW_ENDPOINT, timeout=10)

def trigger_reindex(source_path: str, force: bool):
    payload = {"source_path": source_path, "force_add": force}
    return _make_request("post", ADMIN_REINDEX_ENDPOINT, json=payload, timeout=60)

def clear_user_memory(user_id_to_clear: str):
    return _make_request("delete", f"{ADMIN_CLEAR_MEMORY_ENDPOINT}/{user_id_to_clear}", timeout=15)

def check_health():
    return _make_request("get", ADMIN_HEALTH_ENDPOINT, timeout=15)


# --- Main UI Tabs ---
tab_chat, tab_goals, tab_pipelines, tab_admin = st.tabs(["💬 Chat", "🎯 Goals", "▶️ Pipelines", "⚙️ Admin"])

# --- Chat Tab ---
with tab_chat:
    st.subheader("Chat Interface")
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Chat input
    if prompt := st.chat_input("Ask the agent..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Agent is processing..."):
            api_response = send_chat_message(prompt, st.session_state.user_id)
        response_content = api_response.get("final_output", "Sorry, I encountered an issue.")
        error_content = api_response.get("error")
        with st.chat_message("assistant"):
            st.markdown(response_content)
            if error_content:
                st.error(f"Agent reported an error: {error_content}")
        st.session_state.messages.append({"role": "assistant", "content": response_content})

# --- Goal Management Tab ---
with tab_goals:
    st.subheader("Goal Management")
    col_start, col_status = st.columns(2)

    with col_start:
        st.markdown("#### Start New Goal")
        # TODO: Fetch available goal types from API? For now, manual input.
        goal_type = st.text_input("Goal Type", "website_monitor", help="e.g., website_monitor, data_processing_pipeline")
        goal_desc = st.text_area("Goal Description", "Monitor https://example.com and check up to 5 times.", height=100)
        if st.button("Start Goal"):
            if goal_type and goal_desc:
                with st.spinner("Starting goal..."):
                    result = start_goal(goal_type, goal_desc)
                if result and result.get("goal_id"):
                    st.success(f"Goal started successfully! ID: `{result['goal_id']}`")
                    # Add to session state for tracking
                    st.session_state.active_goals[result['goal_id']] = {"status": "started", "type": goal_type, "description": goal_desc}
                    st.rerun() # Refresh UI to show new goal
                elif result and result.get("error"):
                    st.error(f"Failed to start goal: {result['error']}")
                else:
                    st.error("Failed to start goal. Check API connection/logs.")
            else:
                st.warning("Please provide both Goal Type and Description.")

    with col_status:
        st.markdown("#### Active Goals")
        if not st.session_state.active_goals:
            st.caption("No active goals tracked in this session.")

        goal_ids_to_remove = []
        for goal_id, goal_info in list(st.session_state.active_goals.items()):
            status = goal_info.get("status", "unknown")
            type = goal_info.get("type", "N/A")
            desc = goal_info.get("description", "N/A")[:50] + "..."
            with st.expander(f"`{goal_id}` ({type}) - Status: {status.upper()}", expanded=status=="running"):
                st.caption(f"Description: {desc}")
                col_b1, col_b2, col_b3 = st.columns(3)
                if col_b1.button("Check Status", key=f"check_{goal_id}"):
                    with st.spinner(f"Checking status for {goal_id}..."):
                        status_result = get_goal_status(goal_id)
                    if status_result:
                        st.session_state.active_goals[goal_id].update(status_result) # Update status in session
                        st.rerun() # Refresh UI
                    else:
                        st.warning("Failed to fetch status.")
                if status in ["started", "running"]:
                    if col_b2.button("Cancel Goal", key=f"cancel_{goal_id}"):
                         with st.spinner(f"Cancelling {goal_id}..."):
                              cancel_result = cancel_goal(goal_id)
                         if cancel_result and cancel_result.get("status") in ["cancelled", "cancel_requested"]:
                              st.success(f"Goal {goal_id} cancellation requested/confirmed.")
                              st.session_state.active_goals[goal_id]["status"] = cancel_result["status"]
                              st.rerun()
                         elif cancel_result and cancel_result.get("error"):
                              st.error(f"Error cancelling goal: {cancel_result['error']}")
                         else:
                              st.warning("Failed to cancel goal or get confirmation.")
                if status not in ["started", "running"]:
                     if col_b3.button("Remove from list", key=f"remove_{goal_id}"):
                          goal_ids_to_remove.append(goal_id)

                # Display details from last status check
                if state_snap := goal_info.get("state_snapshot"): st.json(state_snap, expanded=False)
                if state := goal_info.get("state"): st.json(state, expanded=False)
                if final_state := goal_info.get("final_state"): st.json(final_state, expanded=False)
                if error := goal_info.get("error"): st.error(f"Error reported: {error}")

        # Remove goals marked for removal
        if goal_ids_to_remove:
             for goal_id in goal_ids_to_remove:
                  if goal_id in st.session_state.active_goals:
                       del st.session_state.active_goals[goal_id]
             st.rerun()


# --- Pipeline Management Tab ---
with tab_pipelines:
    st.subheader("Pipeline Management")
    col_start_pipe, col_status_pipe = st.columns(2)

    with col_start_pipe:
        st.markdown("#### Start New Pipeline Run")
        # TODO: Fetch available pipeline IDs from API? For now, manual input.
        pipeline_id = st.text_input("Pipeline ID", "csv_analysis_report", help="e.g., csv_analysis_report, simple_sql_summary")
        initial_context_str = st.text_area("Initial Context (JSON)", '{"file_path": "employees.csv"}', height=100, help="Data needed to start the pipeline.")
        if st.button("Start Pipeline Run"):
            if pipeline_id:
                initial_context = None
                try:
                    initial_context = json.loads(initial_context_str) if initial_context_str else {}
                    if not isinstance(initial_context, dict): raise ValueError("Initial context must be a JSON object.")

                    with st.spinner(f"Starting pipeline '{pipeline_id}'..."):
                        result = start_pipeline(pipeline_id, initial_context)

                    if result and result.get("run_id"):
                        st.success(f"Pipeline run started successfully! Run ID: `{result['run_id']}`")
                        st.session_state.active_pipelines[result['run_id']] = {"status": "started", "pipeline_id": pipeline_id, "initial_context": initial_context}
                        st.rerun()
                    elif result and result.get("error"):
                        st.error(f"Failed to start pipeline: {result['error']}")
                    else:
                        st.error("Failed to start pipeline. Check API connection/logs.")

                except json.JSONDecodeError:
                    st.error("Invalid JSON format for Initial Context.")
                except ValueError as ve:
                     st.error(f"Context Error: {ve}")
                except Exception as e:
                     st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Please provide a Pipeline ID.")

    with col_status_pipe:
        st.markdown("#### Active & Recent Pipeline Runs")
        if not st.session_state.active_pipelines:
            st.caption("No pipeline runs tracked in this session.")

        run_ids_to_remove = []
        for run_id, run_info in list(st.session_state.active_pipelines.items()):
            status = run_info.get("status", "unknown")
            p_id = run_info.get("pipeline_id", "N/A")
            with st.expander(f"`{run_id}` ({p_id}) - Status: {status.upper()}", expanded=status=="running"):
                st.caption(f"Initial Context: `{run_info.get('initial_context', {})}`")
                col_pb1, col_pb2, col_pb3 = st.columns(3)
                if col_pb1.button("Check Status", key=f"check_pipe_{run_id}"):
                    with st.spinner(f"Checking status for {run_id}..."):
                        status_result = get_pipeline_status(run_id)
                    if status_result:
                        st.session_state.active_pipelines[run_id].update(status_result)
                        st.rerun()
                    else:
                        st.warning("Failed to fetch status.")
                if status in ["started", "running"]:
                    if col_pb2.button("Cancel Run", key=f"cancel_pipe_{run_id}"):
                         with st.spinner(f"Cancelling {run_id}..."):
                              cancel_result = cancel_pipeline(run_id)
                         if cancel_result and cancel_result.get("status") in ["cancelled", "cancel_requested"]:
                              st.success(f"Pipeline run {run_id} cancellation requested/confirmed.")
                              st.session_state.active_pipelines[run_id]["status"] = cancel_result["status"]
                              st.rerun()
                         elif cancel_result and cancel_result.get("error"):
                              st.error(f"Error cancelling run: {cancel_result['error']}")
                         else:
                              st.warning("Failed to cancel run or get confirmation.")
                if status not in ["started", "running"]:
                     if col_pb3.button("Remove from list", key=f"remove_pipe_{run_id}"):
                          run_ids_to_remove.append(run_id)

                # Display details from last status check
                if result := run_info.get("result"): st.json(result, expanded=False)
                if error := run_info.get("error"): st.error(f"Error reported: {error}")

        # Remove runs marked for removal
        if run_ids_to_remove:
             for run_id in run_ids_to_remove:
                  if run_id in st.session_state.active_pipelines:
                       del st.session_state.active_pipelines[run_id]
             st.rerun()


# --- Admin Tab ---
with tab_admin:
    st.subheader("Admin Actions")
    st.warning("These actions require a valid Admin API Key (see sidebar).")

    # Health Check
    st.markdown("#### System Health")
    if st.button("Check System Health"):
        with st.spinner("Performing health checks..."):
            health_result = check_health()
        if health_result:
            st.metric("Overall Status", health_result.get("status", "UNKNOWN"))
            st.json(health_result.get("details", {}))
        else:
            st.error("Failed to perform health check.")

    st.divider()

    # Knowledge Re-index
    st.markdown("#### Knowledge Base Management")
    reindex_path = st.text_input("Source Path to Re-index", "company_policy.txt", help="Path relative to workspace or absolute.")
    force_reindex = st.checkbox("Force Re-index (ignore memory check)")
    if st.button("Trigger Re-index"):
        if reindex_path:
            with st.spinner(f"Triggering re-index for '{reindex_path}'..."):
                reindex_result = trigger_reindex(reindex_path, force_reindex)
            if reindex_result and reindex_result.get("status") == "success":
                st.success(reindex_result.get("message", "Re-index triggered successfully."))
            elif reindex_result:
                st.error(f"Re-index failed: {reindex_result.get('message', 'Unknown error')}")
            else:
                st.error("Failed to trigger re-index. Check API connection/logs.")
        else:
            st.warning("Please provide a source path.")

    st.divider()

    # Clear Memory
    st.markdown("#### Memory Management")
    user_id_to_clear = st.text_input("User ID Memory to Clear", st.session_state.user_id, help="Enter the user ID whose memory should be cleared.")
    if st.button("Clear User Memory", type="secondary"):
        if user_id_to_clear:
            # Add a confirmation step? This is destructive.
            confirm = st.checkbox(f"Confirm clearing memory for '{user_id_to_clear}'?", value=False, key="confirm_mem_clear")
            if confirm:
                with st.spinner(f"Clearing memory for '{user_id_to_clear}'..."):
                    clear_result = clear_user_memory(user_id_to_clear)
                if clear_result and clear_result.get("status") == "success":
                    st.success(clear_result.get("message", "Memory cleared successfully."))
                elif clear_result:
                    st.error(f"Failed to clear memory: {clear_result.get('message', 'Unknown error')}")
                else:
                    st.error("Failed to clear memory. Check API connection/logs.")
                # Reset checkbox
                st.session_state.confirm_mem_clear = False
                st.rerun()
            elif st.session_state.get("confirm_mem_clear"): # Check if button was clicked but confirm not checked
                 st.warning("Please check the confirmation box to clear memory.")

        else:
            st.warning("Please enter a User ID to clear memory.")

    st.divider()

    # Manual Event Trigger (Moved from main column)
    st.markdown("#### Manual Event Trigger")
    event_type_input = st.text_input("Event Type", "api_trigger", key="admin_event_type")
    event_data_input = st.text_area("Event Data (JSON)", '{"action": "summarize", "target": "latest_sales.csv"}', key="admin_event_data")
    if st.button("Enqueue Event", key="admin_enqueue"):
        try:
            event_data_dict = json.loads(event_data_input) if event_data_input else {}
            enqueue_result = enqueue_event(event_type_input, event_data_dict)
            if enqueue_result:
                st.success(f"Event '{event_type_input}' enqueued successfully!")
            # Error handled by _make_request
        except json.JSONDecodeError:
            st.error("Invalid JSON in Event Data.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")



