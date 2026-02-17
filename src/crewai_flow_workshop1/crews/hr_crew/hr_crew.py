from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FirecrawlSearchTool
from typing import List


@CrewBase
class HrCrew:
    """HR Job Creation crew with market research, AI skills research, and job posting writing."""

    agents: List[BaseAgent]
    tasks: List[Task]
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def job_market_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["job_market_researcher"],  # type: ignore[index]
            tools=[FirecrawlSearchTool()],
            verbose=True,
        )

    @agent
    def ai_skills_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["ai_skills_researcher"],  # type: ignore[index]
            tools=[FirecrawlSearchTool()],
            verbose=True,
        )

    @agent
    def job_posting_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["job_posting_writer"],  # type: ignore[index]
            tools=[FirecrawlSearchTool()],
            verbose=True,
        )

    @task
    def job_market_research_task(self) -> Task:
        return Task(
            config=self.tasks_config["job_market_research_task"],  # type: ignore[index]
            async_execution=True,
        )

    @task
    def ai_skills_research_task(self) -> Task:
        return Task(
            config=self.tasks_config["ai_skills_research_task"],  # type: ignore[index]
            async_execution=True,
        )

    @task
    def job_posting_creation_task(self) -> Task:
        return Task(
            config=self.tasks_config["job_posting_creation_task"],  # type: ignore[index]
            context=[self.job_market_research_task(), self.ai_skills_research_task()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the HrCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
