"""
Face registration + recognition app (Gradio UI + FastAPI + SQLite + DeepFace + Gemini).

Stack: Python · Gradio · DeepFace · SQLite · OpenCV · FastAPI · Google Gemini (LangChain)

Run from `4_langgraph`:
  pip install gradio fastapi uvicorn deepface opencv-python-headless pillow numpy
        langchain-google-genai langchain-core python-dotenv tensorflow
  python app.py

Then open http://127.0.0.1:7860/ui  (Gradio) and http://127.0.0.1:7860/docs (FastAPI).
Set GOOGLE_API_KEY in repo-root `.env` for Gemini summaries.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Env: load before LangChain / tracing tweaks (avoids huge vision payloads in LangSmith)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_ENV = _ROOT / ".env"
load_dotenv(dotenv_path=_ENV, override=False)

if os.getenv("APP_DISABLE_LANGSMITH_FOR_VISION", "true").lower() in ("1", "true", "yes"):
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

import gradio as gr
from deepface import DeepFace
from gradio import mount_gradio_app
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).resolve().parent / "faces_registry.db"
DEEPFACE_MODEL = os.getenv("DEEPFACE_MODEL", "Facenet")
DEEPFACE_DETECTOR = os.getenv("DEEPFACE_DETECTOR", "opencv")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
# Cosine distance from DeepFace; lower = more similar. Typical Facenet cosine < ~0.40 same person.
MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))

_llm: ChatGoogleGenerativeAI | None = None


def _get_gemini() -> ChatGoogleGenerativeAI | None:
    global _llm
    if _llm is not None:
        return _llm
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("Google_API_key")
    if not key:
        return None
    _llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2,
        google_api_key=key,
    )
    return _llm


def _gemini_summary(system_facts: str) -> str:
    llm = _get_gemini()
    if llm is None:
        return "(Gemini skipped: set GOOGLE_API_KEY in .env for an LLM summary.)"
    try:
        msg = HumanMessage(
            content=(
                "You are a concise assistant for a face-recognition demo. "
                "Summarize what happened in 2–4 short sentences for a non-expert user.\n\n"
                f"Facts:\n{system_facts}"
            )
        )
        out = llm.invoke([msg])
        return str(out.content).strip()
    except Exception as e:
        return f"(Gemini error: {type(e).__name__}: {e})"


# ---------------------------------------------------------------------------
# OpenCV + image helpers
# ---------------------------------------------------------------------------
def pil_to_rgb_numpy(image: Any) -> np.ndarray:
    """Gradio PIL / numpy -> RGB uint8 HxWx3."""
    if image is None:
        raise ValueError("No image provided.")
    if hasattr(image, "convert"):
        image = image.convert("RGB")
        return np.array(image, dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr.astype(np.uint8)


def preprocess_image(rgb: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Resize large images with OpenCV to keep DeepFace faster and more stable."""
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    scale = max_side / m
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _represent(rgb: np.ndarray) -> list[float]:
    rgb = preprocess_image(rgb)
    # DeepFace accepts numpy RGB paths; write temp file for maximum compatibility on Windows
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_path, bgr)
        reps = DeepFace.represent(
            img_path=tmp_path,
            model_name=DEEPFACE_MODEL,
            enforce_detection=True,
            detector_backend=DEEPFACE_DETECTOR,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    if not reps:
        raise ValueError("DeepFace returned no embedding.")
    return [float(x) for x in reps[0]["embedding"]]


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------
def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    with db_conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                embedding_json TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )


def db_upsert_face(name: str, embedding: list[float]) -> None:
    payload = json.dumps(embedding)
    with db_conn() as c:
        c.execute(
            """
            INSERT INTO faces (name, embedding_json, model)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                embedding_json = excluded.embedding_json,
                model = excluded.model,
                created_at = datetime('now')
            """,
            (name.strip(), payload, DEEPFACE_MODEL),
        )


def db_all_embeddings() -> list[tuple[str, list[float]]]:
    with db_conn() as c:
        rows = c.execute("SELECT name, embedding_json FROM faces").fetchall()
    out: list[tuple[str, list[float]]] = []
    for r in rows:
        out.append((r["name"], json.loads(r["embedding_json"])))
    return out


# ---------------------------------------------------------------------------
# Face logic (DeepFace + cosine distance)
# ---------------------------------------------------------------------------
def _cosine_distance(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    na = np.linalg.norm(va) or 1.0
    nb = np.linalg.norm(vb) or 1.0
    return float(1.0 - np.dot(va, vb) / (na * nb))


def register_face(name: str, image: Any) -> tuple[str, str]:
    if not name or not str(name).strip():
        return "Please enter a name.", ""
    try:
        rgb = pil_to_rgb_numpy(image)
        emb = _represent(rgb)
    except Exception as e:
        facts = f"Registration failed for name '{name}'. Error: {type(e).__name__}: {e}"
        return facts, _gemini_summary(facts)

    db_init()
    db_upsert_face(str(name).strip(), emb)
    facts = (
        f"Registered face for '{name.strip()}' using DeepFace model {DEEPFACE_MODEL}. "
        f"Embedding length {len(emb)} stored in SQLite at {DB_PATH}."
    )
    return facts, _gemini_summary(facts)


def recognize_face(image: Any) -> tuple[str, str]:
    try:
        rgb = pil_to_rgb_numpy(image)
        query = _represent(rgb)
    except Exception as e:
        facts = f"Recognition failed. Error: {type(e).__name__}: {e}"
        return facts, _gemini_summary(facts)

    db_init()
    rows = db_all_embeddings()
    if not rows:
        facts = "No faces registered yet. Use Register first."
        return facts, _gemini_summary(facts)

    best_name: str | None = None
    best_dist = 1e9
    for name, emb in rows:
        d = _cosine_distance(query, emb)
        if d < best_dist:
            best_dist = d
            best_name = name

    if best_name is None or best_dist > MATCH_THRESHOLD:
        facts = (
            f"No confident match (best distance {best_dist:.4f}, threshold {MATCH_THRESHOLD}). "
            "Try registering this person or use a clearer frontal photo."
        )
    else:
        facts = (
            f"Match: **{best_name}** (cosine distance {best_dist:.4f}; threshold {MATCH_THRESHOLD}). "
            f"Compared against {len(rows)} registered face(s)."
        )
    return facts, _gemini_summary(facts)


# ---------------------------------------------------------------------------
# Gradio
# ---------------------------------------------------------------------------
def _build_gradio() -> gr.Blocks:
    with gr.Blocks(title="Face recognition (DeepFace + Gemini)") as demo:
        gr.Markdown(
            "## Face registration & recognition\n"
            f"**DeepFace** model `{DEEPFACE_MODEL}`, detector `{DEEPFACE_DETECTOR}`. "
            f"**SQLite** `{DB_PATH.name}`. **Gemini** summaries use `{GEMINI_MODEL}` when `GOOGLE_API_KEY` is set."
        )
        img = gr.Image(type="pil", label="Upload face photo")
        name = gr.Textbox(label="Person name (for Register)", placeholder="e.g. Alice")
        out_facts = gr.Markdown(label="Result")
        out_llm = gr.Markdown(label="Gemini summary")

        with gr.Row():
            btn_reg = gr.Button("Register", variant="primary")
            btn_rec = gr.Button("Recognize", variant="secondary")
            btn_clr = gr.Button("Clear")

        btn_reg.click(register_face, inputs=[name, img], outputs=[out_facts, out_llm])
        btn_rec.click(recognize_face, inputs=[img], outputs=[out_facts, out_llm])
        btn_clr.click(
            lambda: ("", "", None, ""),
            inputs=[],
            outputs=[out_facts, out_llm, img, name],
        )

    return demo


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    db_init()
    fastapi_app = FastAPI(
        title="Face Recognition API",
        description="DeepFace + SQLite + optional Gemini summaries. Gradio UI at /ui.",
    )

    @fastapi_app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "db": str(DB_PATH)}

    @fastapi_app.post("/api/register")
    async def api_register(
        name: str = Form(...),
        file: UploadFile = File(...),
    ) -> JSONResponse:
        raw = await file.read()
        pil = None
        try:
            from PIL import Image

            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Invalid image: {e}"}, status_code=400)
        facts, llm = register_face(name, pil)
        return JSONResponse({"ok": True, "facts": facts, "gemini": llm})

    @fastapi_app.post("/api/recognize")
    async def api_recognize(file: UploadFile = File(...)) -> JSONResponse:
        raw = await file.read()
        try:
            from PIL import Image

            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Invalid image: {e}"}, status_code=400)
        facts, llm = recognize_face(pil)
        return JSONResponse({"ok": True, "facts": facts, "gemini": llm})

    gradio_demo = _build_gradio()
    fastapi_app = mount_gradio_app(fastapi_app, gradio_demo, path="/ui")
    return fastapi_app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    # Pass the ASGI app object so startup works regardless of import name (vs "app:app" string).
    uvicorn.run(app, host="127.0.0.1", port=7860, reload=False)
