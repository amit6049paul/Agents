import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, tools

# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure LLM settings for Google Gemini
llm_config = {
    "model": "gemini-1.5-pro",
    "api_key": gemini_api_key,
    "api_type": "google"
}

# Tool 1: Simple calculator
@tools.register
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Tool 2: String manipulation
@tools.register
def reverse_string(text: str) -> str:
    """Reverse a given string."""
    return text[::-1]

# Create User Proxy Agent
user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=5,
)

# Create Assistant Agent
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    tools=[add_numbers, reverse_string]
)

# Start conversation
if __name__ == "__main__":
    user_proxy.initiate_chat(
        assistant,
        message="Can you add 5 and 3? Then reverse the result as a string?"
    )
