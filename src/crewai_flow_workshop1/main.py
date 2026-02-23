#!/usr/bin/env python
from datetime import datetime
from typing import List, Literal, Optional

from crewai import LLM, Agent
from crewai.flow import Flow, listen, persist, router, start
from crewai_tools import FirecrawlScrapeWebsiteTool, FirecrawlSearchTool
from pydantic import BaseModel, Field

# ── Data Models ──────────────────────────────────────────────────────────────


class Message(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class RouterOutput(BaseModel):
    user_intent: Literal["search", "casual_chat"]
    search_queries: Optional[List[str]] = None
    chat_response: Optional[str] = None
    reasoning: str


class FlowState(BaseModel):
    user_message: str = "Hey"
    message_history: List[Message] = []
    search_queries: Optional[List[str]] = None
    chat_response: Optional[str] = None
    response: Optional[str] = None


# ── Flow ─────────────────────────────────────────────────────────────────────


@persist()
class DeepResearchFlow(Flow[FlowState]):
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.state.message_history.append(Message(role=role, content=content))

    # ── 1. Entry point ───────────────────────────────────────────────────

    @start()
    def starting_flow(self):
        self.add_message("user", self.state.user_message)
        return self.state.user_message

    # ── 2. Router — single LLM call for classification + response ────────

    @router(starting_flow)
    def classify_and_respond(self):
        llm = LLM(model="gpt-4.1-mini", temperature=0.1, response_format=RouterOutput)

        prompt = f"""
        <task>
        You are the router for a search assistant. Classify the user's message and
        decide whether to search the web or reply conversationally.
        </task>

        <routing_rules>
        Return "casual_chat" ONLY when the message is:
        - A pure greeting ("hi", "hello", "hey there")
        - A thank-you or farewell ("thanks!", "bye")
        - A meta-question about this assistant ("what can you do?", "how do you work?")
        - Anything that is not a clear research query

        Return "search" for EVERYTHING else, including:
        - Any factual question, no matter how simple
        - Requests for explanations, comparisons, or summaries
        - Vague or broad topics (search anyway — you can refine later)
        - Opinions or recommendations (search for expert perspectives)

        Tie-breaking rule: When in doubt, return "casual_chat" and a follow up question to clarify.
        </routing_rules>

        <output_instructions>
        If intent is "casual_chat":
        - Set chat_response: a brief, friendly reply. If the user seems to want
        information, nudge them to ask a question so you can search for it.

        If intent is "search":
        - Set search_queries: 3-5 concise search queries phrased the way a human
        would type them into a search engine. Cover different angles of the question.

        Always set reasoning: one sentence explaining your decision.
        </output_instructions>

        <inputs>
        Current message: {self.state.user_message}

        Conversation history:
        {self.state.message_history}
        </inputs>
        """

        response = llm.call(prompt)

        if response.user_intent == "casual_chat":
            self.state.chat_response = response.chat_response
            return "casual_chat"

        else:
            self.state.search_queries = response.search_queries
            return "search"

    # ── 3. Chat path — just present what the router generated ────────────

    @listen("casual_chat")
    def present_chat_response(self):
        response = (
            self.state.chat_response
            or "Hey! I'm a search assistant. Ask me anything and I'll look it up for you."
        )
        self.add_message("assistant", response)
        print(f"Assistant: {response}")
        return response

    # ── 4. Search path — inline Agent with Firecrawl tools ───────────────

    @listen("search")
    def execute_search(self):
        print(f"\nSearching for: {self.state.user_message}")
        print(f"Queries: {self.state.search_queries}\n")

        agent = Agent(
            role="Web Search Assistant",
            goal=(
                "Search the web, synthesize what you find, and give the user "
                "a direct, concise answer with numbered source citations."
            ),
            backstory=(
                "You are a search assistant that finds information online and "
                "explains it clearly and concisely. You write like a "
                "knowledgeable friend, not a research paper. Every factual "
                "claim gets a numbered citation."
            ),
            tools=[
                FirecrawlSearchTool(),
                FirecrawlScrapeWebsiteTool(),
            ],
            llm=LLM(model="gpt-4.1-mini", temperature=0.2),
            verbose=True,
        )

        search_prompt = f"""
        <task>
        Answer the user's question by searching the web. Be direct, concise, and cite
        every factual claim with a numbered reference.
        </task>

        <user_question>
        {self.state.user_message}
        </user_question>

        <search_queries>
        Execute these searches using FirecrawlSearchTool:
        {self.state.search_queries}
        </search_queries>

        <tool_instructions>
        Follow this sequence strictly:
        1. Search ALL queries above using the search tool first.
        2. Read the snippets returned by the search results.
        3. Assess: do the snippets contain enough information to answer the question well?
        - If YES → write your answer using the snippet information. Do NOT scrape.
        - If NO → pick the 1-3 most promising URLs and scrape them with
            FirecrawlScrapeWebsiteTool for deeper content. Hard cap: never scrape
            more than 3 URLs total.
        </tool_instructions>

        <planning>
        Before you start searching, briefly note:
        - What you expect to find
        - What would count as a good answer

        After searching (before scraping), briefly note:
        - What the snippets told you
        - Whether you need to scrape for more detail, and which URLs
        </planning>

        <response_format>
        Write your answer following these rules:
        - Start directly with the answer. No preamble like "Based on my research..." or
        "Great question!". Just begin with the substance.
        - Use **bold** for key terms, names, and numbers.
        - Add numbered citations inline like [1], [2] after each factual claim.
        - Write 3-6 paragraphs. No headers or bullet-point lists.
        - Keep a conversational but informative tone.
        - End with a Sources section formatted exactly like this:

        Sources
        [1] Title - URL
        [2] Title - URL
        </response_format>

        <quality_rules>
        - Every factual claim must have a citation. No exceptions.
        - Prefer primary sources (official docs, original papers) over aggregators.
        - If sources disagree, mention the disagreement.
        - If you cannot find reliable information, say so. Never hallucinate URLs.
        - Do not repeat the user's question back to them.
        </quality_rules>
        """

        result = agent.kickoff(search_prompt)
        answer = result.raw

        self.state.response = answer
        self.add_message("assistant", answer)

        print(f"\n{answer}")
        return answer


# ── Entry Points ─────────────────────────────────────────────────────────────


def kickoff():
    deep_research_flow = DeepResearchFlow(tracing=True)
    deep_research_flow.kickoff()


def plot():
    deep_research_flow = DeepResearchFlow()
    deep_research_flow.plot()


if __name__ == "__main__":
    kickoff()
