from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
load_dotenv()



@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


result = add_numbers.invoke({
    "a": 10,
    "b": 20,
})

print(result)
# Initialize the latest Gemini Flash model
model= ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=1.0,
    timeout=None,
    max_retries=2,
)

response = model.invoke(
    "Explain what an AI agent is in simple terms."
)

print(response.content)