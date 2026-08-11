# LangChain + Gemini Agent Learning Project

A beginner-friendly project for learning how to build **AI agents with Python, LangChain, and Google Gemini**.

The goal of this project is to understand the fundamentals by building and running small examples locally in Cursor.

---

## 1. What We Are Learning

This project covers:

* Python virtual environments (`.venv`)
* Environment variables and `.env`
* Calling Gemini from Python
* LangChain
* Chat models
* System and user prompts
* LangChain tools
* Tool calling
* Agents
* Agent loops
* Agent state and memory
* Streaming
* Agent harness concepts
* Error handling
* LangSmith tracing
* Basic agent architecture

The learning progression is:

```text
Python
   ↓
Gemini API
   ↓
LangChain
   ↓
Prompts
   ↓
Tools
   ↓
Tool Calling
   ↓
Agents
   ↓
Agent State / Memory
   ↓
Streaming
   ↓
Agent Harness
   ↓
Tracing / Evaluation
   ↓
Real-world Agent
```

---

# 2. What Is an AI Agent?

A normal LLM application looks like:

```text
User
 ↓
LLM
 ↓
Answer
```

An agent can use tools to perform actions:

```text
User
 ↓
Agent
 ↓
Gemini
 ↓
Does it need a tool?
 ↓
 ┌───────────────┐
 │               │
YES              NO
 │                │
 ▼                ▼
Tool             Answer
 │
 ▼
Tool Result
 │
 ▼
Gemini
 │
 ▼
Final Answer
```

An easy mental model is:

```text
Gemini = Brain

Tools = Hands

Agent = Brain + Hands + Decision Loop

Harness = Everything around the agent
```

---

# 3. What Is LangChain?

**LangChain** is a framework for building applications around language models.

It provides building blocks for:

* Models
* Prompts
* Messages
* Tools
* Agents
* State
* Middleware
* Memory
* Streaming
* Tracing

Instead of manually implementing every part of an agent application, LangChain provides reusable components.

---

# 4. What Is Gemini?

**Gemini** is Google's family of generative AI models.

In this project, Gemini is the LLM that performs the reasoning/generation.

We will access Gemini through LangChain using:

```python
ChatGoogleGenerativeAI
```

---

# 5. Project Setup

## Prerequisites

Install:

* Python 3.10+
* Cursor
* Gemini API key

Check Python:

```bash
python --version
```

---

# 6. Create the Project

Create a folder:

```text
langchain-gemini-agent
```

Open the folder in Cursor.

Recommended initial structure:

```text
langchain-gemini-agent/
│
├── .venv/
├── .env
├── .gitignore
├── requirements.txt
│
├── 01_basic_gemini.py
├── 02_prompt.py
├── 03_tool.py
├── 04_agent.py
├── 05_agent_tools.py
├── 06_memory.py
└── 07_streaming.py
```

---

# 7. Create a Python Virtual Environment

A `.venv` is an isolated Python environment for your project.

It prevents packages from different projects from interfering with each other.

For example:

```text
Project A
└── .venv
    └── Django 4

Project B
└── .venv
    └── Django 5
```

Each project can have its own dependencies.

## Windows

Create the environment:

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

You should see:

```text
(.venv)
```

in your terminal.

## macOS/Linux

Create:

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

---

# 8. Install Dependencies

With `.venv` activated:

```bash
pip install -U langchain langchain-google-genai python-dotenv
```

Save installed dependencies:

```bash
pip freeze > requirements.txt
```

Later, dependencies can be installed with:

```bash
pip install -r requirements.txt
```

---

# 9. Environment Variables

Create:

```text
.env
```

Add your Gemini API key:

```text
GOOGLE_API_KEY=your_actual_api_key
```

Do **not** put your API key directly inside Python code.

Create `.gitignore`:

```text
.venv/
.env
__pycache__/
*.pyc
```

This prevents your API key and virtual environment from accidentally being committed to Git.

---

# 10. First Gemini + LangChain Example

Create:

```text
01_basic_gemini.py
```

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

response = model.invoke(
    "Explain what an AI agent is in simple terms."
)

print(response.content)
```

Run:

```bash
python 01_basic_gemini.py
```

Basic flow:

```text
Python
  ↓
LangChain
  ↓
ChatGoogleGenerativeAI
  ↓
Gemini API
  ↓
Gemini
  ↓
Response
```

---

# 11. Understanding the Model

This:

```python
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)
```

creates the Gemini model interface.

This:

```python
model.invoke(...)
```

sends a request to Gemini.

---

# 12. System and User Prompts

Create:

```text
02_prompt.py
```

Example:

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

messages = [
    (
        "system",
        "You are a Python teacher. "
        "Explain technical concepts to beginners "
        "using simple examples."
    ),
    (
        "human",
        "What is a Python decorator?"
    ),
]

response = model.invoke(messages)

print(response.content)
```

Run:

```bash
python 02_prompt.py
```

## System message

The system message provides instructions to the model.

Example:

```text
You are a Python teacher.
```

## Human/User message

The user message contains the actual request:

```text
Explain Python decorators.
```

Conceptually:

```text
SYSTEM
  ↓
"You are a Python teacher"

USER
  ↓
"Explain decorators"

  ↓
GEMINI

  ↓
ANSWER
```

---

# 13. What Is a Tool?

An LLM normally generates text.

A **tool** gives the agent an ability to interact with something external.

Examples:

```text
Calculator
Database
API
File system
Search
Weather service
Email service
Internal application
```

A tool can simply be a Python function.

---

# 14. Create Your First Tool

Create:

```text
03_tool.py
```

```python
from langchain.tools import tool


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


result = add_numbers.invoke({
    "a": 10,
    "b": 20,
})

print(result)
```

Run:

```bash
python 03_tool.py
```

Output:

```text
30
```

The:

```python
@tool
```

decorator tells LangChain that this function should be treated as a tool.

The function's description is also important:

```python
"""Add two numbers together."""
```

The model uses the tool name, description, and input schema to understand what the tool does.

---

# 15. What Is an Agent?

A basic model call:

```python
model.invoke(...)
```

looks like:

```text
User
 ↓
Gemini
 ↓
Answer
```

An agent adds decision-making and tools:

```text
User
 ↓
Agent
 ↓
Gemini
 ↓
Does Gemini need a tool?
 ↓
YES
 ↓
Tool
 ↓
Tool Result
 ↓
Gemini
 ↓
Final Answer
```

LangChain provides `create_agent()` for creating this type of agent.

---

# 16. Create a Basic Agent

Create:

```text
04_agent.py
```

```python
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
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
```

Run:

```bash
python 04_agent.py
```

---

# 17. Agent With Tools

Now give the agent actual capabilities.

Create:

```text
05_agent_tools.py
```

```python
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
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
```

Run:

```bash
python 05_agent_tools.py
```

The important section is:

```python
agent = create_agent(
    model=model,
    tools=[
        add_numbers,
        multiply_numbers,
    ],
)
```

We are giving the agent:

```text
Gemini
+
Tools
```

---

# 18. Agent Tool-Calling Flow

For:

```text
What is 25 multiplied by 40?
```

the conceptual flow is:

```text
USER
 │
 ▼
AGENT
 │
 ▼
GEMINI
 │
 │ "I need the multiply_numbers tool"
 ▼
multiply_numbers(25, 40)
 │
 ▼
1000
 │
 ▼
GEMINI
 │
 ▼
"The answer is 1000."
```

The important idea is:

> The model decides when it needs a tool, the application executes the tool, and the result is sent back to the model.

---

# 19. A More Realistic Tool

Tools don't have to be calculators.

Example:

```python
@tool
def get_employee_salary(employee_name: str) -> str:
    """Return salary information for an employee."""

    employees = {
        "Alice": "$80,000",
        "Bob": "$95,000",
        "Charlie": "$110,000",
    }

    return employees.get(
        employee_name,
        "Employee not found"
    )
```

An agent could receive:

```text
What is Bob's salary?
```

and use:

```text
get_employee_salary("Bob")
```

The tool returns:

```text
$95,000
```

The agent then produces the final answer.

---

# 20. Do Not Use `eval()` for Calculators

Avoid code such as:

```python
eval(user_input)
```

especially when processing user input.

`eval()` can execute arbitrary Python expressions and can create serious security problems.

Instead, create narrowly scoped tools:

```python
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b
```

---

# 21. Agent Harness

An **agent harness** is the surrounding infrastructure that controls and supports an agent.

A useful mental model:

```text
                 AGENT HARNESS
┌─────────────────────────────────────────┐
│                                         │
│  System Prompt                          │
│                                         │
│  Model                                  │
│                                         │
│  Tools                                  │
│                                         │
│  State                                  │
│                                         │
│  Memory                                 │
│                                         │
│  Middleware                             │
│                                         │
│  Tool Error Handling                    │
│                                         │
│  Guardrails                             │
│                                         │
│  Logging / Tracing                      │
│                                         │
│  Retries                                │
│                                         │
│  Permissions                            │
│                                         │
│  Human Approval                         │
│                                         │
└────────────────────┬────────────────────┘
                     │
                     ▼
                   AGENT
```

The agent itself is only one component.

A production agent normally needs infrastructure around it.

---

# 22. Example: Customer Support Agent

Imagine an agent that handles customer support.

It might have:

```text
                 CUSTOMER SUPPORT AGENT
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
        Search FAQ      Order DB      Refund API
            │              │              │
            └──────────────┼──────────────┘
                           │
                           ▼
                         Gemini
```

The harness might enforce rules such as:

```text
Can this user access the refund API?
        │
       NO
        │
       STOP
```

or:

```text
Refund amount > $500?
        │
       YES
        │
        ▼
Human approval required
```

This is where agent engineering becomes more than prompt engineering.

---

# 23. State and Memory

Agents need state to keep track of information during execution.

For example:

```text
User:
My name is Rahul.

Agent:
Nice to meet you Rahul.

User:
What's my name?

Agent:
Your name is Rahul.
```

Important concepts to learn:

```text
messages
   ↓
state
   ↓
short-term memory
   ↓
long-term memory
```

Start with message history before moving to more complex memory systems.

---

# 24. Streaming

With:

```python
agent.invoke(...)
```

you normally wait for the result.

Streaming allows you to observe execution as it happens.

Basic example:

```python
for chunk in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Calculate 20% of 500."
            }
        ]
    }
):
    print(chunk)
```

Run:

```bash
python 07_streaming.py
```

Streaming becomes particularly useful for:

* Chat applications
* Debugging
* Observing tool calls
* Long-running agents
* User experience

---

# 25. LangSmith

When agents become more complex, `print()` statements are not enough.

You want to see an execution trace like:

```text
Agent
 ├── Gemini call
 ├── Tool call
 │    └── calculator
 ├── Tool result
 ├── Gemini call
 └── Final answer
```

LangSmith can be used for:

* Tracing
* Debugging
* Monitoring
* Evaluation
* Understanding agent execution

Eventually your `.env` can contain:

```text
GOOGLE_API_KEY=your_gemini_key

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
```

Do not commit `.env` to Git.

---

# 26. Recommended Learning Roadmap

Learn the concepts in this order.

## Level 1 — Python

Understand:

```text
Functions
Classes
Decorators
Lists
Dictionaries
JSON
Exceptions
async/await
Environment variables
```

## Level 2 — Gemini

Understand:

```text
API key
Model
Temperature
System instruction
User message
Response
Tokens
Streaming
Structured output
```

## Level 3 — LangChain

Understand:

```text
Chat model
Messages
Prompts
Tools
Tool calling
Agents
```

## Level 4 — Agents

Understand:

```text
create_agent()
tools
agent.invoke()
agent.stream()
system_prompt
tool selection
tool arguments
tool results
```

## Level 5 — Agent Harness

Learn:

```text
Middleware
State
Memory
Guardrails
Tool permissions
Error handling
Retries
Human approval
Logging
Tracing
```

## Level 6 — Production

Eventually learn:

```text
FastAPI
PostgreSQL
Redis
Docker
Authentication
Observability
Evaluation
CI/CD
Deployment
```

---

# 27. Recommended Project Progression

Build these files one at a time:

```text
01_basic_gemini.py
        ↓
02_prompt.py
        ↓
03_tool.py
        ↓
04_agent.py
        ↓
05_agent_tools.py
        ↓
06_memory.py
        ↓
07_streaming.py
        ↓
08_langsmith.py
        ↓
09_real_project/
```

Do not immediately start with a large agent framework.

Run each example, modify it, break it intentionally, and understand what happens.

---

# 28. Mini Project: Personal Research Agent

After completing the basics, build a small agent with three tools:

```text
search_notes()
calculate()
save_note()
```

Architecture:

```text
                   USER
                     │
                     ▼
              ┌──────────────┐
              │ Gemini Agent │
              └──────┬───────┘
                     │
       ┌─────────────┼──────────────┐
       │             │              │
       ▼             ▼              ▼
 search_notes    calculator     save_note
       │             │              │
       └─────────────┼──────────────┘
                     │
                     ▼
                  Answer
```

Example request:

```text
I have 3 projects.

Project A costs $1200.
Project B costs $800.
Project C costs $1500.

What is the total?
Save the result as "project budget".
```

The agent could:

```text
1. Understand the request
2. Calculate the total
3. Save the result
4. Answer the user
```

This teaches the core concepts of agentic applications.

---

# 29. Useful Cursor Commands

## Activate environment

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

## Install package

```bash
pip install package-name
```

## Update packages

```bash
pip install -U langchain langchain-google-genai
```

## Save dependencies

```bash
pip freeze > requirements.txt
```

## Install requirements

```bash
pip install -r requirements.txt
```

## Run Python

```bash
python filename.py
```

## Check Python version

```bash
python --version
```

## Check pip

```bash
pip --version
```

## Deactivate environment

```bash
deactivate
```

---

# 30. Final Mental Model

The most important architecture to remember is:

```text
                     LLM
                      │
               "I can reason"
                      │
                      ▼
                   AGENT
                      │
              "What do I need?"
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        TOOL        TOOL        TOOL
          │           │           │
         API         DB         Python
          │           │           │
          └───────────┼───────────┘
                      ▼
                   RESULT
                      │
                      ▼
                     LLM
                      │
                      ▼
                 FINAL ANSWER
```

And the broader system is:

```text
                    AGENT HARNESS
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
      Gemini             Tools            State
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                     Middleware
                          │
                    Guardrails
                          │
                    Error Handling
                          │
                     Observability
                          │
                       Agent
                          │
                          ▼
                     User / App
```

## Key Takeaways

* **Python** runs your application.
* **Gemini** is the language model.
* **LangChain** provides the application building blocks.
* **Tools** give the model capabilities.
* **Tool calling** lets the model request those capabilities.
* **Agents** manage the model/tool interaction loop.
* **State and memory** let applications maintain context.
* **Harness** refers to the surrounding controls and infrastructure.
* **LangSmith** helps trace and debug agent execution.
* **Guardrails and permissions** become important when agents can perform real actions.

The goal is not to memorize LangChain APIs. The goal is to understand the architecture:

```text
Model
  +
Prompt
  +
Tools
  +
Agent Loop
  +
State
  +
Harness
  =
Agentic Application
```

---

## Useful Documentation

* LangChain Agents: https://docs.langchain.com/oss/python/langchain/agents
* LangChain Tools: https://docs.langchain.com/oss/python/langchain/tools
* Gemini integration: https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai
* LangChain Quickstart: https://docs.langchain.com/oss/python/langchain/quickstart
