# Deep Research Flow

A conversational deep-research system built with [CrewAI Flow](https://crewai.com). It routes between casual chat and web research — when you ask a question worth searching, it generates queries and launches a research agent with Firecrawl to produce a cited answer.

## How It Works

```
User message → @start() starting_flow
    → @router() classify_and_respond (single LLM call)
        → "casual_chat"  → present_chat_response()  (friendly reply)
        → "search"       → execute_search()          (inline Agent + Firecrawl)
```

1. **You send a message** — the router classifies your intent in a single `gpt-4.1-mini` call
2. **Casual chat** — greetings, thank-yous, or meta-questions get a quick conversational reply
3. **Search** — anything factual triggers 3–5 search queries, handed to an inline Agent with Firecrawl tools
4. **The agent searches and scrapes** — using `FirecrawlSearchTool` and `FirecrawlScrapeWebsiteTool`
5. **You get a cited answer** — every factual claim has a numbered inline citation

State persists across runs via `@persist()`, so you can refine your topic over multiple invocations.

## Setup

### Prerequisites

- Python >=3.10, <3.14
- [uv](https://docs.astral.sh/uv/) for dependency management

### Install

```bash
crewai install
```

### Configure API Keys

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

## Running Locally

### Backend (CrewAI Flow)

```bash
crewai run
```

The default user message is set in `FlowState.user_message`. Modify the `kickoff()` function or deploy to CrewAI AMP to accept dynamic input.

### Frontend (Streamlit)

The `research_frontend/` directory contains a Streamlit chat UI that connects to a deployed CrewAI AMP instance.

```bash
cd research_frontend
uv sync
uv run streamlit run app.py
```

Before running, configure the AMP connection. Create `research_frontend/.streamlit/secrets.toml`:

```toml
CRW_API_URL = "https://your-crew.crewai.com"
CRW_API_TOKEN = "your_api_token"
```

The frontend features:
- In-memory multi-chat management (new chats, switch, delete)
- Kickoff/poll pattern against the CrewAI AMP API
- Dark theme with coral accent, CrewAI branding
- Animated thinking indicator while research is in progress

## Deploying the Frontend to Heroku

The Streamlit frontend lives in the `research_frontend/` subdirectory. Heroku deploys it using `git subtree`.

### 1. Create the Heroku app (first time only)

```bash
heroku create your-app-name
```

### 2. Set config vars

```bash
heroku config:set CRW_API_URL=https://your-crew.crewai.com
heroku config:set CRW_API_TOKEN=your_api_token
```

### 3. Deploy

```bash
git subtree push --prefix research_frontend heroku main
```

Heroku picks up `.python-version` and `pyproject.toml` automatically — uv handles dependency installation natively, so no `requirements.txt` is needed. The `Procfile` starts Streamlit on `$PORT`.

## Architecture

### Data Models

| Model | Purpose |
|---|---|
| `Message` | Chat message with role, content, timestamp |
| `RouterOutput` | Structured LLM response: `user_intent` (`search` or `casual_chat`) + search queries or chat response |
| `FlowState` | Persisted state: message history, search queries, chat response, final response |

### Flow Methods

| Method | Decorator | What it does |
|---|---|---|
| `starting_flow` | `@start()` | Appends user message to history |
| `classify_and_respond` | `@router(starting_flow)` | Single LLM call — classifies intent AND generates response/plan |
| `present_chat_response` | `@listen("casual_chat")` | Prints the chat reply from state |
| `execute_search` | `@listen("search")` | Runs inline Agent with Firecrawl tools, produces cited answer |

### Key Design Decisions

- **Single router LLM call** — no separate calls for classification vs response generation
- **Inline Agent** — no Crew overhead for a single-agent research task
- **Firecrawl tools** — `FirecrawlSearchTool` for web search, `FirecrawlScrapeWebsiteTool` for deep page content
- **Mandatory citations** — every factual claim must have an inline source URL
- **`gpt-4.1-mini`** — used for both routing (temperature 0.1) and research (temperature 0.2)

### Project Structure

```
deep_research_template/
├── pyproject.toml              # Backend dependencies (CrewAI flow)
├── .env                        # OPENAI_API_KEY, FIRECRAWL_API_KEY
├── src/
│   └── deep_research_agent/
│       └── main.py             # Flow, state models, router, agent
└── research_frontend/          # Streamlit chat UI (deployed separately)
    ├── pyproject.toml          # Frontend dependencies (streamlit, requests)
    ├── Procfile                # Heroku: streamlit run app.py
    ├── .python-version         # Python 3.13 for Heroku/uv
    ├── .streamlit/
    │   ├── config.toml         # Dark theme config
    │   └── secrets.toml        # CRW_API_URL, CRW_API_TOKEN (local only)
    ├── app.py                  # Chat UI, session state, sidebar
    ├── api.py                  # AMP API client (kickoff/poll)
    └── assets/
        └── crewai_logo.svg     # Branding
```

## Troubleshooting

| Issue | Fix |
|---|---|
| Missing API keys | Ensure `OPENAI_API_KEY` and `FIRECRAWL_API_KEY` are in `.env` |
| Frontend can't connect | Check `CRW_API_URL` and `CRW_API_TOKEN` in `secrets.toml` or Heroku config vars |
| Wrong Python version | Use Python >=3.10, <3.14 (backend) or 3.13 (frontend/Heroku) |
| Dependencies missing | Run `crewai install` (backend) or `uv sync` (frontend) |
| Stale persisted state | Delete the `.crewai` persistence directory and re-run |
| Heroku deploy fails | Make sure you push the subtree: `git subtree push --prefix research_frontend heroku main` |
