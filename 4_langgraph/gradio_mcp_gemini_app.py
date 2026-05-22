"""
Gradio + MCP Server + Gemini LLM Demo
=====================================
Two MCP tools:
  1. read_uploaded_file  – parse a user-uploaded text / PDF / code file
  2. github_file_content – fetch any public (or private) file from GitHub via the REST API

Requirements:
  pip install gradio langchain-google-genai langchain-core mcp requests PyMuPDF python-dotenv

Create a .env file at the repo root before running:
  GOOGLE_API_KEY=your-gemini-key
  GITHUB_TOKEN=ghp_...      # optional – raises rate-limit & enables private repos
  GEMINI_MODEL=gemini-3-flash-preview   # optional
"""

import asyncio
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Optional

import gradio as gr
import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# ── Load .env (repo root, same as app_imagerecognize.py) ─────────────────────
_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_env, override=False)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI | None:
    global _llm
    if _llm is not None:
        return _llm
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("Google_API_key")
    if not key:
        return None
    _llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        google_api_key=key,
    )
    return _llm


def _message_text(content) -> str:
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts).strip()
    return str(content).strip()

# ── MCP server (in-process) ──────────────────────────────────────────────────
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types as mcp_types

mcp_app = Server("gradio-demo-server")


# ════════════════════════════════════════════════════════════════════════════
#  TOOL 1 – read_uploaded_file
# ════════════════════════════════════════════════════════════════════════════
@mcp_app.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(
            name="read_uploaded_file",
            description=(
                "Read the content of a locally uploaded file. "
                "Supports plain text, source code, and PDF files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file on disk.",
                    }
                },
                "required": ["file_path"],
            },
        ),
        mcp_types.Tool(
            name="github_file_content",
            description=(
                "Fetch the raw content of a file (or directory listing) from a "
                "GitHub repository using the GitHub REST API."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "GitHub username or organisation.",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path inside the repo, e.g. 'README.md' or 'src/main.py'.",
                    },
                    "ref": {
                        "type": "string",
                        "description": "Branch, tag, or commit SHA (default: repo default branch).",
                        "default": "HEAD",
                    },
                },
                "required": ["owner", "repo", "path"],
            },
        ),
    ]


@mcp_app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[mcp_types.TextContent]:

    # ── Tool 1 ────────────────────────────────────────────────────────────
    if name == "read_uploaded_file":
        file_path = arguments["file_path"]
        p = Path(file_path)
        if not p.exists():
            return [mcp_types.TextContent(type="text", text=f"ERROR: File not found: {file_path}")]

        suffix = p.suffix.lower()
        try:
            if suffix == ".pdf":
                import fitz  # PyMuPDF
                doc = fitz.open(str(p))
                text = "\n\n".join(page.get_text() for page in doc)
                doc.close()
            else:
                text = p.read_text(errors="replace")
        except Exception as exc:
            return [mcp_types.TextContent(type="text", text=f"ERROR reading file: {exc}")]

        preview = text[:4000] + ("\n…[truncated]" if len(text) > 4000 else "")
        return [mcp_types.TextContent(type="text", text=preview)]

    # ── Tool 2 ────────────────────────────────────────────────────────────
    elif name == "github_file_content":
        owner  = arguments["owner"]
        repo   = arguments["repo"]
        path   = arguments["path"]
        ref    = arguments.get("ref", "HEAD")

        url     = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        headers = {"Accept": "application/vnd.github+json"}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        params  = {"ref": ref}

        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            return [mcp_types.TextContent(
                type="text",
                text=f"GitHub API error {resp.status_code}: {resp.text[:500]}",
            )]

        data = resp.json()

        # Directory listing
        if isinstance(data, list):
            entries = "\n".join(
                f"{'📁' if e['type']=='dir' else '📄'} {e['name']} ({e.get('size',0)} bytes)"
                for e in data
            )
            return [mcp_types.TextContent(type="text", text=f"Directory listing:\n{entries}")]

        # Single file – decode base64
        import base64
        raw = base64.b64decode(data["content"]).decode(errors="replace")
        preview = raw[:4000] + ("\n…[truncated]" if len(raw) > 4000 else "")
        return [mcp_types.TextContent(
            type="text",
            text=f"File: {data['path']}  ({data.get('size',0)} bytes)\n\n{preview}",
        )]

    return [mcp_types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ════════════════════════════════════════════════════════════════════════════
#  MCP helper – call a tool directly (no subprocess needed for in-process use)
# ════════════════════════════════════════════════════════════════════════════
async def _run_tool(tool_name: str, args: dict) -> str:
    results = await call_tool(tool_name, args)
    return "\n".join(r.text for r in results)


def run_tool_sync(tool_name: str, args: dict) -> str:
    """Thread-safe wrapper for the async tool runner."""
    return asyncio.run(_run_tool(tool_name, args))


# ════════════════════════════════════════════════════════════════════════════
#  LangChain tools (wired to MCP implementations above)
# ════════════════════════════════════════════════════════════════════════════
@tool
def read_uploaded_file(file_path: str) -> str:
    """Read the content of a locally uploaded file (text, code, PDF)."""
    return run_tool_sync("read_uploaded_file", {"file_path": file_path})


@tool
def github_file_content(
    owner: str, repo: str, path: str, ref: str = "HEAD"
) -> str:
    """Fetch a file or directory from a GitHub repository via the REST API."""
    return run_tool_sync(
        "github_file_content",
        {"owner": owner, "repo": repo, "path": path, "ref": ref},
    )


_AGENT_TOOLS = [read_uploaded_file, github_file_content]
_TOOL_BY_NAME = {t.name: t for t in _AGENT_TOOLS}


def gemini_agent(user_message: str, uploaded_file_path: Optional[str]) -> str:
    """
    Run a Gemini agent (ChatGoogleGenerativeAI) that can call our two MCP tools.
    Returns the final text answer.
    """
    llm = _get_llm()
    if llm is None:
        return (
            "Missing **`GOOGLE_API_KEY`** (or `Google_API_key`) in `.env`. "
            "Add it at the repo root, same as `app_imagerecognize.py`."
        )

    system = (
        "You are a helpful assistant with access to two tools:\n"
        "1. read_uploaded_file – use this when the user wants you to analyse "
        "   the file they uploaded (the absolute path is given in the message).\n"
        "2. github_file_content – use this to fetch files from GitHub.\n"
        "Always call a tool when the user's request requires reading external content. "
        "Summarise tool output clearly for the user."
    )

    if uploaded_file_path:
        user_message = (
            f"{user_message}\n\n[Uploaded file is at: {uploaded_file_path}]"
        )

    llm_with_tools = llm.bind_tools(_AGENT_TOOLS)
    messages: list = [
        SystemMessage(content=system),
        HumanMessage(content=user_message),
    ]

    try:
        response: AIMessage = llm_with_tools.invoke(messages)
        MAX_TURNS = 5
        for _ in range(MAX_TURNS):
            if not response.tool_calls:
                break
            messages.append(response)
            for tc in response.tool_calls:
                tool_fn = _TOOL_BY_NAME.get(tc["name"])
                if tool_fn is None:
                    result = f"Unknown tool: {tc['name']}"
                else:
                    result = tool_fn.invoke(tc["args"])
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )
            response = llm_with_tools.invoke(messages)

        return _message_text(response.content) or "_(no text response from model)_"

    except Exception as e:
        return (
            f"**Gemini error:** `{type(e).__name__}: {e}`\n\n"
            "Check `GOOGLE_API_KEY`, model name (`GEMINI_MODEL`), and quota at Google AI / Gemini API."
        )


# ════════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ════════════════════════════════════════════════════════════════════════════
_uploaded_path: dict = {}   # simple shared state (one user demo)


def handle_upload(file) -> str:
    if file is None:
        return "No file uploaded."
    _uploaded_path["path"] = file.name
    return f"✅ File ready: `{Path(file.name).name}`"


def handle_chat(user_msg: str, history: list) -> tuple[list, str]:
    fpath = _uploaded_path.get("path")
    answer = gemini_agent(user_msg, fpath)
    history = history or []
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": answer})
    return history, ""


def handle_github_fetch(owner: str, repo: str, path: str, ref: str) -> str:
    if not owner or not repo or not path:
        return "Please fill in Owner, Repo, and Path."
    result = run_tool_sync(
        "github_file_content",
        {"owner": owner, "repo": repo, "path": path, "ref": ref or "HEAD"},
    )
    return result


CSS = """
body { font-family: 'JetBrains Mono', monospace; }
.gr-button-primary { background: #1a1a2e !important; }
"""

with gr.Blocks(title="Gradio · MCP · Gemini") as demo:
    gr.Markdown(
        f"""
# 🤖 Gradio + MCP Server + Gemini LLM

Two **MCP tools** wired to **Gemini** (`{GEMINI_MODEL}`):

| Tool | What it does |
|---|---|
| `read_uploaded_file` | Parse a text / code / PDF file you upload |
| `github_file_content` | Fetch any file from a GitHub repo via REST API |
        """
    )

    with gr.Tabs():

        # ── Tab 1 : Chat with file upload ───────────────────────────────
        with gr.Tab("💬 Chat + File Upload"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1 · Upload a file")
                    file_input  = gr.File(
                        label="Upload file (txt / py / pdf / csv / …)",
                        file_types=[".txt", ".py", ".pdf", ".csv", ".json",
                                    ".md", ".js", ".ts", ".java", ".cpp"],
                    )
                    upload_status = gr.Textbox(
                        label="Upload status", interactive=False, lines=1
                    )
                    file_input.change(handle_upload, inputs=file_input, outputs=upload_status)

                    gr.Markdown("### 2 · Ask Gemini about it")
                    gr.Markdown(
                        "_Example prompts_\n"
                        "- `Summarise this file`\n"
                        "- `Find all TODO comments`\n"
                        "- `What are the main functions?`"
                    )

                with gr.Column(scale=2):
                    chatbot  = gr.Chatbot(label="Conversation", height=420)
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Ask Gemini about the uploaded file …",
                            label="Your message",
                            scale=5,
                        )
                        send_btn = gr.Button("Send ➤", variant="primary", scale=1)

                    send_btn.click(
                        handle_chat,
                        inputs=[chat_input, chatbot],
                        outputs=[chatbot, chat_input],
                    )
                    chat_input.submit(
                        handle_chat,
                        inputs=[chat_input, chatbot],
                        outputs=[chatbot, chat_input],
                    )

        # ── Tab 2 : GitHub explorer ──────────────────────────────────────
        with gr.Tab("🐙 GitHub Explorer"):
            gr.Markdown(
                "Fetch any file or directory from GitHub via the REST API. "
                "Set `GITHUB_TOKEN` env var to access private repos and avoid rate limits."
            )
            with gr.Row():
                owner_box = gr.Textbox(label="Owner / Org", placeholder="e.g. python")
                repo_box  = gr.Textbox(label="Repository",  placeholder="e.g. cpython")
                ref_box   = gr.Textbox(label="Branch / Tag / SHA", placeholder="main")
            path_box  = gr.Textbox(label="File / Directory path", placeholder="README.rst")
            fetch_btn = gr.Button("Fetch from GitHub 🔍", variant="primary")
            gh_output = gr.Code(label="Result", language="markdown", lines=24)

            fetch_btn.click(
                handle_github_fetch,
                inputs=[owner_box, repo_box, path_box, ref_box],
                outputs=gh_output,
            )

            gr.Examples(
                examples=[
                    ["python",     "cpython",    "README.rst",       "main"],
                    ["gradio-app", "gradio",     "README.md",        "main"],
                    ["pallets",    "flask",      "src/flask",        "main"],
                    ["tiangolo",   "fastapi",    "pyproject.toml",   "master"],
                ],
                inputs=[owner_box, repo_box, path_box, ref_box],
                label="Quick examples",
            )

        # ── Tab 3 : Setup guide ─────────────────────────────────────────
        with gr.Tab("⚙️ Setup"):
            gr.Markdown(
                """
## Installation

```bash
pip install gradio langchain-google-genai langchain-core mcp requests PyMuPDF python-dotenv
```

## .env file

Create a `.env` file at the **repo root** (same as `app_imagerecognize.py`):

```ini
# Required – get your key at https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=AIza...

# Optional – raises rate limits from 60 → 5 000 req/hr & enables private repos
GITHUB_TOKEN=ghp_...

# Optional – default matches app_imagerecognize.py
GEMINI_MODEL=gemini-3-flash-preview
```

> ⚠️ Add `.env` to your `.gitignore` — never commit API keys.

## Run

```bash
python gradio_mcp_gemini_app.py
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Gradio UI                        │
│  ┌──────────────┐        ┌─────────────────────┐   │
│  │  File Upload │        │   GitHub Explorer   │   │
│  └──────┬───────┘        └────────┬────────────┘   │
│         │                         │                 │
│         └──────────┬──────────────┘                 │
│                    ▼                                 │
│           Gemini (GEMINI_MODEL)  ◄── function calling │
│                    │                                 │
│                    ▼                                 │
│         ┌─────────────────────┐                     │
│         │   MCP Server        │                     │
│         │  (in-process)       │                     │
│         │  ┌───────────────┐  │                     │
│         │  │read_uploaded_ │  │                     │
│         │  │file           │  │                     │
│         │  ├───────────────┤  │                     │
│         │  │github_file_   │  │                     │
│         │  │content        │  │                     │
│         │  └───────────────┘  │                     │
│         └─────────────────────┘                     │
└─────────────────────────────────────────────────────┘
```
                """
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                css=CSS, theme=gr.themes.Soft())