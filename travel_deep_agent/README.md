# Travel Deep Agent (Gemini) — a beginner-friendly example

A small, real, runnable Python project that shows how a "deep agent" system
fits together, using a travel-planning example anyone can follow:

> *"Plan a 5-day trip to Goa for 2 people, budget ₹60,000, into beaches,
> nightlife, and seafood, flying from Mumbai."*

It uses **Google Gemini** (via `google-generativeai`) as the reasoning
engine, reads the API key from a `.env` file, and demonstrates every piece
you asked about:

| Concept | Where it lives | What it does |
|---|---|---|
| **Deep agent** | `agents/orchestrator.py` (`DeepAgent`) | Plans, delegates, accumulates, remembers — instead of one giant prompt |
| **MCP-style tools** | `mcp/tools.py`, `mcp/mcp_server.py` | Typed tool functions registered in a local tool server, callable by name |
| **Task decomposition** | `DeepAgent.decompose()` | Gemini breaks the goal into an ordered JSON task list |
| **Subagents** | `agents/subagents.py` | One specialist per job: Research, Flights, Hotels, Budget, Itinerary |
| **Long-term memory** | `memory/long_term_memory.py` | SQLite DB of traveler preferences + past trips, persists between runs |
| **Context management** | `context/context_manager.py` | Shares findings between subagents; auto-summarizes when it gets long |
| **Skills** | `skills/*/SKILL.md` | Plain-markdown instructions each subagent follows (editable, no code) |

## How it flows

```
 user goal
    │
    ▼
 DeepAgent.decompose()  ──▶  Gemini returns an ordered plan:
    │                        [research, flights, hotels, budget, itinerary]
    ▼
 for each step:
    SubAgent(skill + tools).run(task, shared_context, memory)
        │                 │
        │                 └─ MCPToolServer.call via Gemini automatic
        │                    function calling (search_flights, etc.)
        ▼
    ContextManager.add(output)   ← every step sees what came before
    │
    ▼
 LongTermMemory.save_trip()   ← remembered for next time
    │
    ▼
 final itinerary printed
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste a free key from https://aistudio.google.com/app/apikey
python main.py
```

Running it prints:
1. the registered MCP tools,
2. the planner's task-decomposition JSON,
3. each subagent's output (with its tool calls happening behind the scenes),
4. the final day-by-day itinerary,
5. a note that the trip was saved to `data/travel_memory.db`.

Run it a second time and the Research/Budget agents will have access to
`traveler_1`'s remembered preferences and trip history automatically.

## Why the tools are "mocked"

`mcp/tools.py` returns deterministic, made-up flight/hotel/weather data
instead of calling real travel APIs. This keeps the example:
- runnable with **only** a Gemini key (no Amadeus/Skyscanner/weather API
  keys to collect),
- **repeatable** for learning (same inputs → same outputs),
- easy to swap: replace the body of any function in `mcp/tools.py` with a
  real API call and nothing else in the project needs to change, because
  every other layer only depends on the function's name + typed signature.

## Extending toward real MCP

This project simulates MCP's *shape* (typed tools, discoverable by a
server, called by name) without the JSON-RPC wire protocol, to keep the demo
dependency-free. To go further:
- install the official `mcp` python package,
- start an MCP server process exposing `mcp/tools.py`'s functions,
- replace `MCPToolServer` in `agents/subagents.py` with an MCP client that
  connects to it.

Nothing in `agents/orchestrator.py` or the `skills/` folder would need to
change, since they only depend on `list_tools()` / tool objects, not on how
those tools are transported.

## Project layout

```
travel_deep_agent/
├── main.py                     # run this
├── config.py                   # loads .env
├── agents/
│   ├── orchestrator.py         # DeepAgent: decompose → dispatch → remember
│   └── subagents.py            # SubAgent wrapper (skill + tools + Gemini)
├── mcp/
│   ├── tools.py                # the actual tool functions (mock travel data)
│   └── mcp_server.py           # local tool registry, MCP-style
├── skills/
│   ├── destination_research/SKILL.md
│   ├── flight_search/SKILL.md
│   ├── hotel_search/SKILL.md
│   ├── budget_planning/SKILL.md
│   └── itinerary_builder/SKILL.md
├── memory/
│   └── long_term_memory.py     # SQLite: preferences + past trips
├── context/
│   └── context_manager.py      # shared working memory + auto-summarization
└── data/                       # travel_memory.db created here at runtime
```

## Ideas to try next

- Add a new subagent (e.g. `visa_requirements`) with its own `SKILL.md` and
  a `check_visa_requirements` tool — the orchestrator's planner will start
  routing to it automatically once it's registered.
- Swap `search_flights`/`search_hotels` for real APIs.
- Add a `packing_list` skill that reads the Research agent's weather output
  from the shared context and produces a checklist.
