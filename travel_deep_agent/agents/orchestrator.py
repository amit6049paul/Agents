"""
agents/orchestrator.py
------------------------
The DEEP AGENT itself. "Deep" here means it doesn't answer in one shot --
it plans, delegates to specialists, and only then composes a final answer:

    user goal
        |
        v
    1) DECOMPOSE  -> Gemini turns the goal into an ordered list of subtasks,
                      each tagged with which subagent should handle it
        |
        v
    2) DISPATCH   -> each subtask is handed to its SubAgent (agents/subagents.py),
                      which uses its skill + tools (mcp/) to do that one job
        |
        v
    3) ACCUMULATE -> every subagent's output is appended to the shared
                      ContextManager so later subagents can build on earlier
                      findings (e.g. Budget agent reuses Flight agent's price)
        |
        v
    4) REMEMBER   -> once done, a short summary is written to long-term
                      memory (memory/) so future runs for this traveler can
                      reference past trips and preferences
"""
import json
import re

import google.generativeai as genai

from config import GEMINI_MODEL
from context.context_manager import ContextManager
from memory.long_term_memory import LongTermMemory
from mcp.mcp_server import MCPToolServer
from mcp.tools import ALL_TOOLS
from agents.subagents import build_subagents

VALID_AGENTS = ["research", "flights", "hotels", "budget", "itinerary"]

PLANNER_PROMPT = """You are the planning module of a travel deep-agent system.
Break the traveler's goal below into an ORDERED JSON list of subtasks.
Each subtask must be an object with:
  "agent": one of {agents}
  "task": one concrete sentence describing exactly what that agent should do,
          including all relevant details (dates, cities, budget, interests,
          passenger count) pulled from the goal.

Always include "research", "flights", "hotels", and "budget" before
"itinerary" (itinerary must be last, since it summarizes everything else).

Respond with ONLY the JSON array, no prose, no markdown fences.

Traveler goal:
{goal}
"""


class DeepAgent:
    def __init__(self):
        self.mcp_server = MCPToolServer()
        self.mcp_server.register_many(ALL_TOOLS)
        self.subagents = build_subagents(self.mcp_server)
        self.memory = LongTermMemory()
        self.planner_model = genai.GenerativeModel(GEMINI_MODEL)

    # ---- step 1: task decomposition -----------------------------------
    def decompose(self, goal: str) -> list:
        prompt = PLANNER_PROMPT.format(agents=VALID_AGENTS, goal=goal)
        response = self.planner_model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
        plan = json.loads(raw)
        for step in plan:
            if step["agent"] not in VALID_AGENTS:
                raise ValueError(f"Planner produced unknown agent: {step['agent']}")
        return plan

    # ---- steps 2-3: dispatch + accumulate context ----------------------
    def run(self, user_id: str, goal: str) -> dict:
        memory_briefing = self.memory.memory_briefing(user_id)
        context = ContextManager()
        context.add("traveler", goal)

        plan = self.decompose(goal)
        step_outputs = {}

        for step in plan:
            agent = self.subagents[step["agent"]]
            output = agent.run(
                task=step["task"],
                shared_context=context.get_context(model=self.planner_model),
                memory_briefing=memory_briefing,
            )
            step_outputs[step["agent"]] = output
            context.add(agent.name, output)

        # ---- step 4: write a short summary to long-term memory --------
        self.memory.save_trip(user_id, {
            "goal": goal,
            "itinerary_excerpt": step_outputs.get("itinerary", "")[:400],
        })

        return {"plan": plan, "steps": step_outputs,
                "final_itinerary": step_outputs.get("itinerary", "")}
