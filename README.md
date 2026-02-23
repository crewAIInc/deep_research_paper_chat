# Deep Research Flow

A conversational deep-research system built with [CrewAI Flow](https://crewai.com). It routes between conversation and research modes — asking clarifying questions until the topic is clear, then executing comprehensive web research with verifiable source citations using Firecrawl.

## How It Works

```
User message → @start() starting_flow
    → @router() classify_and_respond (single LLM call)
        → "conversation"       → present_followup()     (prints follow-up question)
        → "ready_to_research"  → execute_research()      (inline Agent + Firecrawl)
```

1. **You send a message** — the router analyzes your intent in a single LLM call
2. **If your topic is vague** — it asks clarifying questions to narrow scope
3. **If your topic is clear** — it generates search queries and launches a research agent
4. **The agent searches and scrapes** — using FirecrawlSearchTool and FirecrawlScrapeWebsiteTool
5. **You get a report** — with inline source citations on every claim

State persists across runs via `@persist()`, so you can refine your topic over multiple `crewai run` invocations.

## Setup

### Prerequisites

- Python >=3.10, <3.14
- [UV](https://docs.astral.sh/uv/) for dependency management

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

## Usage

```bash
crewai run
```

The default user message is empty. Set it in `FlowState.user_message` or modify the `kickoff()` function to accept input.

### Example Conversation Flow

```
Run 1 — User: "I want to research AI"
         Bot:  "That's a broad topic! Are you interested in recent LLM advances,
                AI in a specific industry, AI policy, or something else?"

Run 2 — User: "Focus on AI agents and tool use in 2024-2025"
         Bot:  → Routes to research
              → Searches 5 queries across different angles
              → Scrapes top sources
              → Returns a structured report with inline citations
```

## Architecture

### Data Models

| Model | Purpose |
|---|---|
| `Message` | Chat message with role, content, timestamp |
| `RouterOutput` | Structured LLM response: intent + follow-up OR research plan |
| `FlowState` | Persisted state: history, topic, queries, report |

### Flow Methods

| Method | Decorator | What it does |
|---|---|---|
| `starting_flow` | `@start()` | Appends user message to history |
| `classify_and_respond` | `@router(starting_flow)` | Single LLM call — classifies intent AND generates response/plan |
| `present_followup` | `@listen("conversation")` | Prints the follow-up question from state |
| `execute_research` | `@listen("ready_to_research")` | Runs inline Agent with Firecrawl tools |

### Key Design Decisions

- **Single router LLM call** — no separate calls for classification vs response generation
- **Inline Agent** — no Crew overhead for a single-agent research task
- **Firecrawl tools** — `FirecrawlSearchTool` for web search, `FirecrawlScrapeWebsiteTool` for deep page content
- **Mandatory citations** — every piece of information must have an inline source URL

## Customization

The research agent prompt in `execute_research()` is marked with a `TODO` comment. This is where you define:

- How the agent should prioritize and evaluate sources
- When to scrape full pages vs rely on search snippets
- The exact output format for the citation-backed report

## Troubleshooting

| Issue | Fix |
|---|---|
| Missing API keys | Ensure `OPENAI_API_KEY` and `FIRECRAWL_API_KEY` are in `.env` |
| Wrong Python version | Use Python >=3.10, <3.14 |
| Dependencies missing | Run `crewai install` |
| Stale persisted state | Delete the `.crewai` persistence directory and re-run |
