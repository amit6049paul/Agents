import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path, override=False)

api_key = os.getenv("OLLAMA_API_KEY")
if not api_key:
    raise ValueError("Missing OLLAMA_API_KEY")

model = ChatOllama(
    model="gpt-oss:120b",  # or another model available in your Ollama Cloud account
    base_url="https://ollama.com",
    client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"



agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in kolkata?"}]}
)
print(result["messages"][-1].content_blocks)
