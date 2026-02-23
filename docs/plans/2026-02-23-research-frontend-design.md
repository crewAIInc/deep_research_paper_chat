# Deep Research Frontend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Streamlit chat UI in `research_frontend/` that connects to the Deep Research flow deployed on CrewAI AMP, deployable to Heroku.

**Architecture:** In-memory multi-chat Streamlit app with no database. All state in `st.session_state` (dict of chats). API client calls CrewAI AMP's kickoff/poll endpoints. Reuses ~80-95% of existing `chat_frontend/` code with research-oriented branding.

**Tech Stack:** Streamlit, requests, Heroku (Procfile-based), CrewAI AMP API

---

### Task 1: Create directory structure and static files

**Files:**
- Create: `research_frontend/assets/crewai_logo.svg` (copy from `chat_frontend/assets/`)
- Create: `research_frontend/.streamlit/config.toml` (copy from `chat_frontend/.streamlit/`)
- Create: `research_frontend/.streamlit/secrets.toml` (placeholder, will be gitignored)
- Create: `research_frontend/.gitignore`

**Step 1: Create directory structure**

```bash
mkdir -p research_frontend/assets research_frontend/.streamlit
```

**Step 2: Copy static assets**

```bash
cp chat_frontend/assets/crewai_logo.svg research_frontend/assets/
cp chat_frontend/.streamlit/config.toml research_frontend/.streamlit/
```

**Step 3: Create .streamlit/secrets.toml placeholder**

```toml
CRW_API_URL = "https://YOUR-DEPLOYMENT-URL.crewai.com"
CRW_API_TOKEN = "YOUR-TOKEN-HERE"
```

**Step 4: Create .gitignore**

```
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info
.venv
.claude/
.streamlit/secrets.toml
.DS_Store
```

**Step 5: Commit**

```bash
git add research_frontend/assets/ research_frontend/.streamlit/config.toml research_frontend/.gitignore
git commit -m "feat: scaffold research_frontend directory with assets and config"
```

---

### Task 2: Create api.py — CrewAI AMP client

**Files:**
- Create: `research_frontend/api.py`
- Reference: `chat_frontend/api.py` (lines 1-78)

**Step 1: Create api.py**

Adapted from `chat_frontend/api.py`. Key change: `_extract_response()` handles both `response` (search path) and `chat_response` (casual_chat path) from the flow's state.

```python
"""CrewAI AMP API client — kickoff and poll for Deep Research responses."""

import json

import requests
import streamlit as st


def _api_url() -> str:
    return st.secrets["CRW_API_URL"].rstrip("/")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {st.secrets['CRW_API_TOKEN']}",
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
```

**Step 2: Verify the file is syntactically correct**

```bash
cd research_frontend && python -c "import ast; ast.parse(open('api.py').read()); print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add research_frontend/api.py
git commit -m "feat: add CrewAI AMP API client for research frontend"
```

---

### Task 3: Create app.py — Main Streamlit UI

**Files:**
- Create: `research_frontend/app.py`
- Reference: `chat_frontend/app.py` (lines 1-523)

This is the largest task. Key differences from `chat_frontend/app.py`:
- **No db.py** — all state in `st.session_state["chats"]` dict
- **No history view** — sidebar is the only navigation
- **Research branding** — headline, subtitle, suggestion chips
- **`extract_response()`** — uses the new helper from api.py

**Step 1: Create app.py**

```python
"""Deep Research Frontend — Streamlit UI with in-memory chat history."""

import base64
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from api import kickoff_research, poll_status, extract_response

# ── Page config ─────────────────────────────────────────────

st.set_page_config(
    page_title="Deep Research Assistant",
    page_icon="assets/crewai_logo.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load logo as base64 for inline embedding ────────────────

LOGO_PATH = Path("assets/crewai_logo.svg")
LOGO_B64 = base64.b64encode(LOGO_PATH.read_bytes()).decode() if LOGO_PATH.exists() else ""

# ── CSS Design System ───────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Custom Properties ────────────────────── */
:root {
    --bg-primary: #0E1117;
    --bg-secondary: #141820;
    --bg-card: #1A1D24;
    --border-subtle: #2A2D35;
    --border-hover: #FF5A50;
    --accent: #FF5A50;
    --accent-dim: rgba(255, 90, 80, 0.08);
    --accent-glow: rgba(255, 90, 80, 0.15);
    --text-primary: #FAFAFA;
    --text-secondary: #8B8D97;
    --text-muted: #6B6E76;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Global ────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: var(--font-family) !important;
}
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary);
}

/* ── Hide Streamlit Branding ──────────────── */
#MainMenu, header[data-testid="stHeader"], footer,
div[data-testid="stDecoration"] {
    display: none !important;
}

/* ── Sidebar ───────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border-subtle);
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
}

/* ── Chat Bubbles ─────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: var(--radius-md);
    margin-bottom: 8px;
    padding: 12px 16px;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--accent-dim);
    border: 1px solid rgba(255, 90, 80, 0.12);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
}

/* ── Thinking Indicator ──────────────────── */
@keyframes thinking-dots {
    0%, 20% { opacity: 0.2; }
    50% { opacity: 1; }
    80%, 100% { opacity: 0.2; }
}
.thinking-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--accent);
    margin: 0 3px;
    animation: thinking-dots 1.4s infinite ease-in-out;
}
.thinking-dot:nth-child(1) { animation-delay: 0s; }
.thinking-dot:nth-child(2) { animation-delay: 0.2s; }
.thinking-dot:nth-child(3) { animation-delay: 0.4s; }
.thinking-container {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
}
.thinking-text {
    color: var(--text-secondary);
    font-family: var(--font-family);
    font-size: 14px;
}

/* ── Empty State ──────────────────────────── */
.empty-state {
    text-align: center;
    padding: 100px 20px 40px 20px;
    color: var(--text-muted);
}
.empty-state img {
    height: 48px;
    margin-bottom: 24px;
}
.empty-state h2 {
    font-family: var(--font-family);
    color: var(--text-primary);
    font-weight: 600;
    font-size: 24px;
    margin-bottom: 8px;
}
.empty-state p {
    font-family: var(--font-family);
    color: var(--text-secondary);
    font-size: 15px;
    margin-bottom: 32px;
}

/* ── Logo ──────────────────────────────────── */
.logo-sidebar {
    text-align: center;
    padding: 12px 0 4px 0;
}
.logo-sidebar img {
    height: 28px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ──────────────────────────────────────

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ── In-memory chat helpers ──────────────────────────────────

def _make_title(text: str, max_len: int = 50) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def _create_chat(chat_id: str, title: str) -> None:
    st.session_state.chats[chat_id] = {
        "title": title,
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }


def _add_message(chat_id: str, role: str, content: str) -> None:
    st.session_state.chats[chat_id]["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    })


def _get_messages(chat_id: str) -> list[dict]:
    chat = st.session_state.chats.get(chat_id)
    return chat["messages"] if chat else []


def _get_sorted_chats() -> list[tuple[str, dict]]:
    return sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True,
    )


def _new_chat() -> None:
    st.session_state.active_chat_id = None
    st.session_state.processing = False


def _switch_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id
    st.session_state.processing = False


def _delete_chat(chat_id: str) -> None:
    st.session_state.chats.pop(chat_id, None)
    if st.session_state.active_chat_id == chat_id:
        st.session_state.active_chat_id = None


# ── Crew response handler ──────────────────────────────────

def _handle_crew_response(chat_id: str, prompt: str, status_area) -> str | None:
    kickoff_id = kickoff_research(prompt, chat_id)
    if not kickoff_id:
        return None

    max_attempts = 150  # 150 x 2s = 5 minutes
    for _ in range(max_attempts):
        status = poll_status(kickoff_id)

        if status["state"] == "SUCCESS" and status["result"]:
            return extract_response(status["result"])

        if status["state"] in ("FAILURE", "TIMEOUT", "ERROR"):
            return None

        task_text = status.get("last_executed_task") or "Researching..."
        status_area.markdown(
            f'<div class="thinking-container">'
            f'<div><span class="thinking-dot"></span>'
            f'<span class="thinking-dot"></span>'
            f'<span class="thinking-dot"></span></div>'
            f'<span class="thinking-text">{task_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        time.sleep(2)

    return None


# ── Render: Empty State ─────────────────────────────────────

SUGGESTIONS = [
    "What are the latest advances in quantum computing?",
    "Compare React vs Vue for large-scale apps",
    "Explain how mRNA vaccines work",
    "What are the best practices for LLM fine-tuning?",
]


def _render_empty_state() -> None:
    logo_html = f'<img src="data:image/svg+xml;base64,{LOGO_B64}">' if LOGO_B64 else ""
    st.markdown(
        f"""
        <div class="empty-state">
            {logo_html}
            <h2>Deep Research Assistant</h2>
            <p>Ask any question and I'll search the web to find a comprehensive answer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 1, 1, 1])
    for i, suggestion in enumerate(SUGGESTIONS):
        with cols[i % 4]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_prompt = suggestion
                st.rerun()


# ── Render: Chat View ───────────────────────────────────────

def _render_chat_view() -> None:
    active_id = st.session_state.active_chat_id
    messages = _get_messages(active_id) if active_id else []

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not active_id and not messages:
        _render_empty_state()

    prompt = None
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    else:
        prompt = st.chat_input(
            "Ask a research question...",
            disabled=st.session_state.processing,
        )

    if prompt:
        if not active_id:
            active_id = str(uuid.uuid4())
            _create_chat(active_id, _make_title(prompt))
            st.session_state.active_chat_id = active_id

        _add_message(active_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.processing = True

        with st.chat_message("assistant"):
            status_area = st.empty()
            status_area.markdown(
                '<div class="thinking-container">'
                '<div><span class="thinking-dot"></span>'
                '<span class="thinking-dot"></span>'
                '<span class="thinking-dot"></span></div>'
                '<span class="thinking-text">Researching...</span>'
                '</div>',
                unsafe_allow_html=True,
            )

            response_text = _handle_crew_response(active_id, prompt, status_area)

            status_area.empty()
            if response_text:
                st.markdown(response_text)
                _add_message(active_id, "assistant", response_text)
            else:
                st.error("Failed to get a response. Please try again.")

        st.session_state.processing = False
        st.rerun()


# ── Sidebar ─────────────────────────────────────────────────

with st.sidebar:
    if LOGO_B64:
        st.markdown(
            f'<div class="logo-sidebar"><img src="data:image/svg+xml;base64,{LOGO_B64}"></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")

    if st.button("+ New Research", type="primary", use_container_width=True):
        _new_chat()
        st.rerun()

    st.caption("RECENT CHATS")

    for chat_id, chat_data in _get_sorted_chats():
        col_btn, col_del = st.columns([5, 1])
        with col_btn:
            is_active = chat_id == st.session_state.active_chat_id
            label = f"{'> ' if is_active else ''}{chat_data['title']}"
            if st.button(label, key=f"chat_{chat_id}", use_container_width=True):
                _switch_chat(chat_id)
                st.rerun()
        with col_del:
            if st.button("x", key=f"del_{chat_id}"):
                _delete_chat(chat_id)
                st.rerun()


# ── Main Content ────────────────────────────────────────────

_render_chat_view()
```

**Step 2: Verify the file is syntactically correct**

```bash
cd research_frontend && python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add research_frontend/app.py
git commit -m "feat: add Streamlit chat UI for Deep Research frontend"
```

---

### Task 4: Create deployment files (pyproject.toml, Procfile, requirements.txt, runtime.txt)

**Files:**
- Create: `research_frontend/pyproject.toml`
- Create: `research_frontend/requirements.txt`
- Create: `research_frontend/Procfile`
- Create: `research_frontend/runtime.txt`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "research-frontend"
version = "0.1.0"
description = "Streamlit chat UI for the Deep Research CrewAI flow"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "requests>=2.32.5",
    "streamlit>=1.54.0",
]
```

**Step 2: Create requirements.txt (for Heroku)**

```
requests>=2.32.5
streamlit>=1.54.0
```

**Step 3: Create Procfile**

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

**Step 4: Create runtime.txt**

```
python-3.13.5
```

**Step 5: Commit**

```bash
git add research_frontend/pyproject.toml research_frontend/requirements.txt research_frontend/Procfile research_frontend/runtime.txt
git commit -m "feat: add Heroku deployment config for research frontend"
```

---

### Task 5: Initialize uv environment and verify local run

**Step 1: Initialize uv in research_frontend**

```bash
cd research_frontend && uv sync
```

**Step 2: Verify the app starts locally**

```bash
cd research_frontend && uv run streamlit run app.py --server.port=8502
```

Expected: App starts on `http://localhost:8502` without import errors. The app will show the empty state with "Deep Research Assistant" headline and 4 suggestion chips. API calls will fail (expected — no real AMP deployment configured yet).

**Step 3: Stop the server (Ctrl+C) and commit any lock file**

```bash
git add research_frontend/uv.lock research_frontend/.python-version
git commit -m "chore: add uv lock file for research frontend"
```

---

### Task 6: Final review and summary commit

**Step 1: Verify all files exist**

```bash
ls -la research_frontend/
ls -la research_frontend/assets/
ls -la research_frontend/.streamlit/
```

Expected file list:
- `app.py`, `api.py`, `pyproject.toml`, `requirements.txt`, `Procfile`, `runtime.txt`, `.gitignore`
- `assets/crewai_logo.svg`
- `.streamlit/config.toml`

**Step 2: Verify .gitignore is working (secrets.toml not tracked)**

```bash
cd research_frontend && git status
```

Expected: `secrets.toml` does NOT appear in untracked files.
