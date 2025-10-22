#!/usr/bin/env python

import json
from datetime import datetime
from typing import List, Literal, Optional

from crewai import LLM, Agent
from crewai.flow import Flow, listen, persist, router, start
from pydantic import BaseModel, Field

from crewai_flow_workshop1.tools.deep_research_paper import (
    DeepResearchPaper,  # Using the local tool
)

# from deep_research_paper_tool.tool import DeepResearchPaper # Importing tool from crewai tool repository


class Message(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class RouterIntent(BaseModel):
    user_intent: Literal["research", "conversation"]
    research_query: Optional[str] = None
    reasoning: str


class Source(BaseModel):
    url: str
    title: str
    relevant_content: str


class SearchResult(BaseModel):
    research_summary: str = Field(
        description="A comprehensive research summary that combines all found sources with inline URL citations. "
        "Each piece of information must be followed by its source URL in parentheses format: (URL). "
        "The summary should be organized by topics/themes and present a cohesive narrative."
    )
    sources_list: List[Source] = Field(
        description="Complete list of all sources used in the research summary for reference."
    )


class FlowState(BaseModel):
    user_message: str = "help me researching on the latest trends in ai"
    message_history: List[Message] = []
    research_query: Optional[str] = None
    user_intent: Optional[Literal["research", "conversation"]] = None
    search_result: Optional[SearchResult] = None


@persist()
class DeepResearchFlow(Flow[FlowState]):
    def add_message(self, role: str, content: str):
        """Add a message to the message history"""
        new_message = Message(role=role, content=content)
        self.state.message_history.append(new_message)

    @start()
    def starting_flow(self):
        # Add the user message to history
        if self.state.user_message:
            self.add_message("user", self.state.user_message)

        return self.state.user_message

    @router(starting_flow)
    def routing_intent(self):
        llm = LLM(model="gpt-4.1-mini", temperature=0.1, response_format=RouterIntent)

        prompt = f"""
        === TASK ===
        You are an intelligent router that determines user intent and generates research queries when appropriate.

        === INSTRUCTIONS ===
        Analyze the user's message and conversation history to determine the intent. Follow these specific rules:

        **RESEARCH Intent Criteria:**
        - User asks for factual information, studies, or analysis that requires external research
        - User asks follow-up questions about previous research topics mentioned in conversation history
        - User requests investigation into scientific papers, market trends, or data analysis
        - User wants in-depth exploration of complex topics that benefit from comprehensive research

        **CONVERSATION Intent Criteria:**
        - General greetings, casual conversation, or personal questions
        - Simple clarifications that don't require external research
        - Requests for explanations about the system's capabilities
        - Any interaction that is purely conversational in nature

        === INPUT DATA ===
        **Current User Message:**
        {self.state.user_message}

        **Recent Conversation History:**
        {self.state.message_history}

        === OUTPUT REQUIREMENTS ===
        1. **user_intent**: Must be either "research" or "conversation"
        2. **research_query**:
        - If intent is "research": Generate a comprehensive, specific research query that incorporates context from conversation history and current message
        - If intent is "conversation": Set to null
        3. **reasoning**: Provide clear reasoning for your classification decision

        === EXAMPLES ===
        **Research Example:**
        User: "What are the latest studies on climate change impacts?"
        → user_intent: "research", research_query: "latest scientific studies climate change environmental impacts 2024", reasoning: "User requests current research information requiring external sources"

        **Conversation Example:**
        User: "Hello, how are you today?"
        → user_intent: "conversation", research_query: null, reasoning: "General greeting requiring conversational response, no research needed"

        **Follow-up Research Example:**
        Previous context: Discussion about AI ethics
        User: "Can you find more studies about bias in machine learning?"
        → user_intent: "research", research_query: "bias machine learning algorithms studies fairness AI ethics research", reasoning: "Follow-up question on previous AI ethics discussion requiring additional research"

        === CRITICAL REQUIREMENTS ===
        - Be decisive and clear in your classification
        - For research queries, make them specific and actionable for a research agent
        - Consider conversation context when generating research queries
        - Always provide reasoning for your decision"""

        response = llm.call(prompt)

        self.state.research_query = response.research_query
        self.state.user_intent = response.user_intent

        return response.user_intent

    @listen("conversation")
    def follow_up_conversation(self):
        llm = LLM(model="gpt-4.1-mini", temperature=0.2)

        prompt = f"""
        === ROLE ===
        You are a helpful and knowledgeable conversation assistant specialized in guiding users toward valuable research opportunities when appropriate.

        === TASK ===
        Provide a natural, helpful response to the user's message while being mindful of conversation flow and context.

        === CONTEXT ===
        **Current User Message:**
        {self.state.user_message}

        **Recent Conversation History:**
        {self.state.message_history}

        === INSTRUCTIONS ===
        1. **Respond naturally**: Address the user's message directly and conversationally
        2. **Be helpful**: Provide useful information or assistance based on their request
        3. **Guide appropriately**: If the conversation naturally leads toward topics that would benefit from research, gently suggest research options
        4. **Maintain context**: Reference previous conversation points when relevant
        5. **Stay engaging**: Keep the conversation flowing and be personable

        === GUIDANCE OPPORTUNITIES ===
        When appropriate, you may suggest research on:
        - Scientific papers and studies
        - Market trends and analysis
        - Technical deep-dives
        - Data-driven insights
        - Current events and developments

        === RESPONSE STYLE ===
        - Be conversational and natural
        - Show understanding of the user's needs
        - Provide actionable next steps when relevant
        - Keep responses concise but comprehensive
        - Be encouraging and supportive

        Respond to the user's message now:"""

        response = llm.call(prompt)

        # Add the conversation response to history
        self.add_message("assistant", response)

        print(f"Conversation response: {response}")
        return response

    @listen("research")
    def handle_research(self):
        print(f"Starting research with query: {self.state.research_query}")

        # Create an Agent for deep research
        analyst = Agent(
            role="Deep Research Specialist",
            goal="Conduct comprehensive research on specific queries, returning a summary response and detailed sources data",
            backstory="You are an expert researcher with access to academic databases and research sources. "
            "You excel at finding relevant scholarly papers, studies, and research findings, "
            "synthesizing multiple academic sources, and providing comprehensive insights from credible research.",
            tools=[DeepResearchPaper()],
            verbose=True,
        )

        # Execute the research
        task = f"""
        You must research and provide comprehensive information about the query:{self.state.research_query}

        OUTPUT FORMAT REQUIREMENTS:
        - Write a comprehensive summary that combines ALL found sources into a single, cohesive narrative
        - Each piece of information MUST be immediately followed by its source URL in parentheses: (https://example.com/source)
        - sources_list: Include ALL sources used, with url, title, and relevant_content for each. Every fact, finding, or piece of information must be cited with its URL.

        <example>
        Example:
        "According to recent research, AI adoption is increasing rapidly (https://example.com/source1), while challenges remain in implementation (https://example.com/source2)."
        </example>
        """

        research_result = analyst.kickoff(task, response_format=SearchResult)

        self.state.search_result = research_result.pydantic

        # Add the research result to conversation history
        self.add_message("assistant", self.state.search_result.research_summary)

        return self.state.model_dump()


def kickoff():
    research_flow = DeepResearchFlow(tracing=True)
    research_flow.kickoff()


def plot():
    research_flow = DeepResearchFlow()
    research_flow.plot()


if __name__ == "__main__":
    kickoff()
