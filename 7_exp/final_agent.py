from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool


load_dotenv()


# -------------------------
# Tools
# -------------------------

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def calculate_percentage(
    number: float,
    percentage: float,
) -> float:
    """Calculate a percentage of a number."""
    return number * percentage / 100


# -------------------------
# Model
# -------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)


# -------------------------
# Agent
# -------------------------

agent = create_agent(
    model=model,
    tools=[
        add_numbers,
        multiply_numbers,
        calculate_percentage,
    ],
    system_prompt="""
You are a helpful mathematical assistant.

Use the available tools whenever calculations
are required.

Explain the final answer clearly.
""",
)


# -------------------------
# Run agent
# -------------------------

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "What is 15% of 800, "
                    "and then multiply the result by 3?"
                ),
            }
        ]
    }
)


# -------------------------
# Print final response
# -------------------------

print("\nAgent response:")
print(result["messages"][-1].content)