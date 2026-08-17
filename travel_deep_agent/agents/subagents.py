"""
agents/subagents.py
---------------------
A SubAgent = one narrow specialist:
    a SKILL (system prompt, loaded from skills/*/SKILL.md)
  + a small TOOLBOX (a subset of mcp/tools.py, via the MCPToolServer)
  + a Gemini model that is allowed to call those tools automatically.

The DeepAgent orchestrator (agents/orchestrator.py) never calls tools or
Gemini directly for the actual work -- it only decides WHICH subagent
handles WHICH subtask, then reads back that subagent's text result. This
separation is what makes it a "deep agent" rather than one giant prompt:
each specialist has a small, focused job and a small, focused toolbox.
"""
import google.generativeai as genai

from config import GEMINI_MODEL
from skills.skill_loader import load_skill


class SubAgent:
    def __init__(self, name: str, skill_name: str, tool_names: list, mcp_server):
        self.name = name
        self.system_prompt = load_skill(skill_name)
        tools = mcp_server.list_tools(tool_names)
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=self.system_prompt,
            tools=tools,
        )

    def run(self, task: str, shared_context: str, memory_briefing: str) -> str:
        chat = self.model.start_chat(enable_automatic_function_calling=True)
        prompt = (
            f"MEMORY ABOUT THIS TRAVELER:\n{memory_briefing}\n\n"
            f"SHARED CONTEXT SO FAR:\n{shared_context}\n\n"
            f"YOUR TASK:\n{task}"
        )
        response = chat.send_message(prompt)
        return response.text.strip()


def build_subagents(mcp_server) -> dict:
    """Wire up one SubAgent per specialty. This mapping is what the
    orchestrator's task decomposition plan refers to by name."""
    return {
        "research": SubAgent(
            "ResearchAgent", "destination_research",
            ["get_weather", "get_attractions"], mcp_server),
        "flights": SubAgent(
            "FlightAgent", "flight_search",
            ["search_flights"], mcp_server),
        "hotels": SubAgent(
            "HotelAgent", "hotel_search",
            ["search_hotels"], mcp_server),
        "budget": SubAgent(
            "BudgetAgent", "budget_planning",
            ["calculate_budget", "convert_currency"], mcp_server),
        "itinerary": SubAgent(
            "ItineraryAgent", "itinerary_builder",
            [], mcp_server),  # synthesis only, no tools needed
    }
