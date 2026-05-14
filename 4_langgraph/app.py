"""
Gradio UI: describe / recognize an image using Google Gemini (same stack as AmitLangGraphTool.py).

Set GOOGLE_API_KEY in repo-root `.env` (optional fallback: Google_API_key).
Optional env: GEMINI_MODEL (default matches project: gemini-3-flash-preview).

LangSmith: full-image traces exceed the ~20MB multipart limit. This file disables
tracing for the process unless APP_DISABLE_LANGSMITH_FOR_VISION=false in .env.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env before LangChain imports so tracing flags apply correctly.
_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_env, override=False)

# Vision payloads embed large base64; LangSmith rejects runs > ~20MB.
if os.getenv("APP_DISABLE_LANGSMITH_FOR_VISION", "true").lower() in ("1", "true", "yes"):
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

import gradio as gr
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

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


def _pil_to_data_uri(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def recognize_image(image, question: str) -> str:
    if image is None:
        return "Please upload an image first."

    llm = _get_llm()
    if llm is None:
        return (
            "Missing **`GOOGLE_API_KEY`** (or `Google_API_key`) in `.env`. "
            "Add it at the repo root, same as `AmitLangGraphTool.py`."
        )

    prompt = (question or "").strip()
    if not prompt:
        prompt = (
            "Describe this image in detail: main objects, people, setting, colors, "
            "and any visible text. If something is unclear, say so."
        )

    try:
        uri = _pil_to_data_uri(image)
        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": uri}},
            ]
        )
        response = llm.invoke([msg])
        text = response.content
        if isinstance(text, list):
            parts = []
            for block in text:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
            return "\n".join(parts).strip()
        return str(text).strip()

    except Exception as e:
        return (
            f"**Gemini error:** `{type(e).__name__}: {e}`\n\n"
            "Check `GOOGLE_API_KEY`, model name (`GEMINI_MODEL`), and quota at Google AI / Gemini API."
        )


def clear_analysis():
    """Reset image, prompt, and analysis output for a new run."""
    return None, "", ""


with gr.Blocks(title="Image recognition (Gemini)") as demo:
    gr.Markdown(
        f"## Recognize an image (**Gemini** — `{GEMINI_MODEL}`)\n"
        "Uses **`GOOGLE_API_KEY`** from `.env` (same pattern as `AmitLangGraphTool.py`). "
        "Optional: set **`GEMINI_MODEL`** in `.env` to override the default model."
    )
    with gr.Row():
        img = gr.Image(type="pil", label="Upload image")
        question = gr.Textbox(
            label="Prompt (optional)",
            placeholder='e.g. "What brand is the logo?"  Leave empty for a full description.',
            lines=3,
        )
    out = gr.Markdown()
    with gr.Row():
        btn = gr.Button("Analyze", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")

    btn.click(recognize_image, inputs=[img, question], outputs=out)
    clear_btn.click(clear_analysis, inputs=[], outputs=[img, question, out])

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
