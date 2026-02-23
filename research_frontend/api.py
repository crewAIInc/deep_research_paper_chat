"""CrewAI AMP API client — kickoff and poll for Deep Research responses."""

import json
import os

import requests
import streamlit as st


def _get_secret(key: str) -> str | None:
    """Read a secret from st.secrets (local) or os.environ (Heroku)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key)


def _api_url() -> str:
    url = _get_secret("CRW_API_URL")
    if not url:
        st.error("CRW_API_URL is not configured. Set it in .streamlit/secrets.toml or as an environment variable.")
        st.stop()
    return url.rstrip("/")


def _headers() -> dict:
    token = _get_secret("CRW_API_TOKEN")
    if not token:
        st.error("CRW_API_TOKEN is not configured. Set it in .streamlit/secrets.toml or as an environment variable.")
        st.stop()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def api_request(endpoint: str, method: str = "GET", data: dict | None = None) -> dict | None:
    """Make an authenticated request to the CrewAI AMP API."""
    url = f"{_api_url()}/{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, headers=_headers(), timeout=30)
        elif method == "POST":
            resp = requests.post(url, headers=_headers(), json=data, timeout=30)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None


def kickoff_research(user_message: str, chat_id: str) -> str | None:
    """Send a message to the Deep Research flow and return the kickoff_id."""
    payload = {
        "inputs": {
            "user_message": user_message,
            "id": chat_id,
        }
    }
    result = api_request("kickoff", method="POST", data=payload)
    if result and "kickoff_id" in result:
        return result["kickoff_id"]
    return None


def poll_status(kickoff_id: str) -> dict:
    """Poll the status endpoint and return the parsed status dict.

    Returns a dict with keys:
        state: str  — "PENDING", "RUNNING", "SUCCESS", "FAILURE", "TIMEOUT"
        result: dict | None — parsed result when state is SUCCESS
        last_executed_task: str | None — description of current progress
    """
    data = api_request(f"status/{kickoff_id}")
    if not data:
        return {"state": "ERROR", "result": None, "last_executed_task": None}

    state = data.get("state", "UNKNOWN")
    last_task = data.get("last_executed_task")

    parsed_result = None
    if state == "SUCCESS" and data.get("result"):
        raw = data["result"]
        if isinstance(raw, str):
            try:
                parsed_result = json.loads(raw)
            except json.JSONDecodeError:
                parsed_result = {"response": raw}
        elif isinstance(raw, dict):
            parsed_result = raw

    return {"state": state, "result": parsed_result, "last_executed_task": last_task}


def extract_response(status_result: dict | None) -> str | None:
    """Extract the response text from a successful status result.

    The Deep Research flow returns 'response' from the search path
    and 'chat_response' from the casual_chat path.
    """
    if not status_result:
        return None
    return status_result.get("response") or status_result.get("chat_response")
