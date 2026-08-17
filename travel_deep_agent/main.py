"""
main.py
--------
Run this file to see the whole system work end to end:

    python main.py

It plans a real (mock-data) trip for a beginner-friendly example:
"5 days in Goa for 2 people, on a 60,000 INR budget, into beaches,
nightlife, and seafood, flying out of Mumbai."

Every piece described in the README (deep agent, MCP-style tools, task
decomposition, subagents, long-term memory, context management, skills) is
exercised by this one call to DeepAgent.run().
"""
import google.generativeai as genai

import config
from agents.orchestrator import DeepAgent

EXAMPLE_GOAL = (
    "Plan a 5-day, 4-night trip to Goa for 2 adults, flying out of Mumbai, "
    "departing 2026-11-10. Total budget is 60,000 INR. We're interested in "
    "beaches, nightlife, and seafood."
)


def main():
    config.require_api_key()
    genai.configure(api_key=config.GEMINI_API_KEY)

    agent = DeepAgent()
    print("Registered MCP tools:\n" + agent.mcp_server.schema_summary() + "\n")

    user_id = "traveler_1"

    # First run: remember a preference for this traveler, so later runs
    # (and later subagents, via memory_briefing) can use it.
    agent.memory.set_preference(user_id, "home_currency", "INR")
    agent.memory.set_preference(user_id, "travel_style", "budget-conscious, loves seafood")

    print(f"Goal:\n{EXAMPLE_GOAL}\n")
    print("Planning trip... (this calls Gemini several times, please wait)\n")

    result = agent.run(user_id, EXAMPLE_GOAL)

    print("=" * 70)
    print("TASK DECOMPOSITION PLAN")
    print("=" * 70)
    for i, step in enumerate(result["plan"], 1):
        print(f"{i}. [{step['agent']}] {step['task']}")

    for agent_name, output in result["steps"].items():
        print("\n" + "=" * 70)
        print(f"{agent_name.upper()} AGENT OUTPUT")
        print("=" * 70)
        print(output)

    print("\n" + "=" * 70)
    print("FINAL ITINERARY")
    print("=" * 70)
    print(result["final_itinerary"])

    print("\n(This trip summary was also saved to long-term memory at "
          f"{config.DB_PATH} -- run main.py again and traveler_1's "
          "preferences will already be known.)")


if __name__ == "__main__":
    main()
