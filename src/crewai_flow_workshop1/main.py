#!/usr/bin/env python
from datetime import datetime
from typing import List, Literal, Optional

from crewai import Agent, LLM
from crewai.flow import Flow, listen, persist, router, start
from pydantic import BaseModel, Field
from crewai_tools import FirecrawlSearchTool

from crewai_flow_workshop1.crews.hr_crew.hr_crew import HrCrew


class Message(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class RouterIntent(BaseModel):
    user_intent: Literal["job_creation", "conversation", "refinement"]
    job_role: Optional[str] = None
    location: Optional[str] = None
    company_name: Optional[str] = None
    feedback: Optional[str] = None
    reasoning: str


class FlowState(BaseModel):
    user_message: str = "HI, create a job posting for a data engineer based in NYC for Johnson & Johnson"
    message_history: List[Message] = []
    job_role: Optional[str] = None
    location: Optional[str] = None
    company_name: Optional[str] = None
    job_posting: Optional[str] = None
    feedback: Optional[str] = None


@persist()
class HrJobCreationFlow(Flow[FlowState]):
    def add_message(self, role: str, content: str):
        """Add a message to the message history"""
        new_message = Message(role=role, content=content)
        self.state.message_history.append(new_message)

    @start()
    def starting_flow(self):
        if self.state.user_message:
            self.add_message("user", self.state.user_message)
        return self.state.user_message

    @router(starting_flow)
    def routing_intent(self):
        llm = LLM(model="gpt-4.1-nano", response_format=RouterIntent)

        has_posting = self.state.job_posting is not None

        prompt = f"""
        === TASK ===
        You are an intelligent router for an HR job creation assistant. Your job is to analyze
        the user's message and conversation history to extract job creation details and determine intent.

        === INSTRUCTIONS ===
        Extract any of these fields mentioned in the current message or conversation history:
        - **job_role**: The job title/role being created (e.g., "Software Engineer", "Marketing Manager")
        - **location**: The job location (e.g., "New York", "Remote", "London")
        - **company_name**: The company the job is for (e.g., "Google", "Acme Corp")

        **ALREADY COLLECTED VALUES (preserve these — do NOT set to null):**
        - job_role: {self.state.job_role or "Not yet collected"}
        - location: {self.state.location or "Not yet collected"}
        - company_name: {self.state.company_name or "Not yet collected"}

        **EXISTING JOB POSTING:** {"Yes — a posting has already been generated" if has_posting else "No posting yet"}

        **ROUTING RULES:**
        - Return "refinement" if a job posting ALREADY EXISTS and the user is giving feedback,
          requesting changes, or asking for improvements to the current posting.
          Also extract the **feedback** field: a concise summary of what the user wants changed.
        - Return "job_creation" if ALL THREE fields (job_role, location, company_name) are populated
          AND no job posting exists yet.
          Also return "job_creation" if the user wants a completely NEW posting for a different
          role/company (even if a posting exists — this resets state).
        - Return "conversation" if any field is still missing and no posting exists yet.

        === INPUT DATA ===
        **Current User Message:**
        {self.state.user_message}

        **Conversation History:**
        {self.state.message_history}

        === OUTPUT REQUIREMENTS ===
        1. **user_intent**: "job_creation", "conversation", or "refinement"
        2. **job_role**: The job role if mentioned (or from already collected values)
        3. **location**: The location if mentioned (or from already collected values)
        4. **company_name**: The company name if mentioned (or from already collected values)
        5. **feedback**: If intent is "refinement", a concise summary of the requested changes. Otherwise null.
        6. **reasoning**: Brief explanation of your decision
        """

        response = llm.call(prompt)

        # Accumulate state — only update fields that are newly extracted (don't overwrite with None)
        if response.job_role:
            self.state.job_role = response.job_role
        if response.location:
            self.state.location = response.location
        if response.company_name:
            self.state.company_name = response.company_name
        if response.feedback:
            self.state.feedback = response.feedback

        return response.user_intent

    @listen("conversation")
    def follow_up_conversation(self):
        llm = LLM(model="gpt-5-nano")

        collected = []
        missing = []

        if self.state.job_role:
            collected.append(f"Job Role: {self.state.job_role}")
        else:
            missing.append("job role")

        if self.state.location:
            collected.append(f"Location: {self.state.location}")
        else:
            missing.append("location")

        if self.state.company_name:
            collected.append(f"Company: {self.state.company_name}")
        else:
            missing.append("company name")

        collected_str = ", ".join(collected) if collected else "Nothing yet"
        missing_str = ", ".join(missing)

        prompt = f"""
        === ROLE ===
        You are a friendly HR assistant helping users create job postings. Your goal is to
        collect three pieces of information: job role, location, and company name.

        === CONTEXT ===
        **Current User Message:**
        {self.state.user_message}

        **Conversation History:**
        {self.state.message_history}

        **Already Collected:**
        {collected_str}

        **Still Needed:**
        {missing_str}

        === INSTRUCTIONS ===
        1. Respond naturally to the user's message
        2. Acknowledge any information they've already provided
        3. Ask for the missing information in a conversational way
        4. Be warm, professional, and concise
        5. If the user hasn't mentioned anything about job creation yet, introduce yourself and
           explain that you can help create job postings

        Respond to the user now:"""

        response = llm.call(prompt)

        self.add_message("assistant", response)
        print(f"Assistant: {response}")
        return response

    @listen("job_creation")
    def handle_job_creation(self):
        print(
            f"\nCreating job posting for: {self.state.job_role} "
            f"at {self.state.company_name} in {self.state.location}\n"
        )

        crew = HrCrew().crew()
        result = crew.kickoff(
            inputs={
                "job_role": self.state.job_role,
                "location": self.state.location,
                "company_name": self.state.company_name,
            }
        )

        response = result.raw
        self.state.job_posting = response

        # Add the conversation response to history
        self.add_message("assistant", response)

        print(f"Conversation response: {response}")
        return response

    @listen("refinement")
    def handle_refinement(self):
        print(f"\nRefining job posting based on feedback: {self.state.feedback}\n")

        llm = LLM(model="gpt-5-nano", temperature=0.3)

        agent = Agent(
            role="Senior Job Posting Editor",
            goal="Refine and improve job postings based on specific feedback",
            backstory=(
                "You are an expert HR editor who specializes in polishing "
                "and refining job postings. You make precise, targeted changes "
                "based on feedback while preserving the overall quality and "
                "structure of the posting."
            ),
            tools=[FirecrawlSearchTool()],
            llm=llm,
            verbose=True,
        )

        prompt = f"""
        You have a job posting that needs refinement based on user feedback.

        === CURRENT JOB POSTING ===
        {self.state.job_posting}

        === USER FEEDBACK ===
        {self.state.feedback}

        === INSTRUCTIONS ===
        - Make ONLY the changes requested in the feedback
        - Preserve the overall structure and quality of the posting
        - Keep all sections that aren't affected by the feedback
        - Return the complete updated job posting in markdown format
        """

        result = agent.kickoff(prompt)
        response = result.raw
        self.state.job_posting = response

        # Add the conversation response to history
        self.add_message("assistant", response)

        print(f"Conversation response: {response}")
        return response


def kickoff():
    hr_flow = HrJobCreationFlow(tracing=True)
    hr_flow.kickoff()


def plot():
    hr_flow = HrJobCreationFlow()
    hr_flow.plot()


if __name__ == "__main__":
    kickoff()
