from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


model= ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=1.0,
    timeout=None,
    max_retries=2,
)


agent = create_agent(
    model=model,
    tools=[
        add_numbers,
        multiply_numbers,
    ],
    system_prompt=(
        "You are a helpful math assistant. "
        "Use the available tools when appropriate."
    ),
)


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 25 multiplied by 40?"
            }
        ]
    }
)


print(result["messages"][-1].content)