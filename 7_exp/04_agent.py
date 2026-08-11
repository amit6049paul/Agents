from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

load_dotenv()

model= ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=1.0,
    timeout=None,
    max_retries=2,
)

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a helpful Python teacher.",
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain Python virtual environments."
            }
        ]
    }
)

print(result["messages"][-1].content)