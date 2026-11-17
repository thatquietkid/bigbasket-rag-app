"""
api.py — FastAPI bridge between the React voice orb and the existing backend.

The orb records in the browser, posts a WAV here, and this module reuses the
project's real STT + RAG code (stt.py, rag.py). No second implementation of the
assistant is introduced.

Run:
    uvicorn api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from rag import RAGPipeline, TOP_K, live_request_kwargs
from stt import RealtimeTranscriber, TranscriptionError, wav_to_pcm16_24k

app = FastAPI(
    title="BigBasket Voice API",
    version="1.0.0",
    description="Voice transcription and catalogue-grounded answers for the React orb.",
)

# Vite proxies /api to this server, but allow direct access too (e.g. preview).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Load the retriever once; subsequent questions share the warm index."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def voice_key() -> str:
    return os.getenv("OPENAI_API", "") or os.getenv("OPENAI_API_KEY", "")


class AskBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=1200)
    history: List[Dict[str, str]] = Field(default_factory=list)
    k: int = Field(default=TOP_K, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(dense|bm25|hybrid)$")
    live: bool = False
    platform: str = "BigBasket"
    lat: Optional[float] = None
    lon: Optional[float] = None
    pincode: Optional[str] = None


def _transcribe(pcm: bytes, key: str) -> str:
    with RealtimeTranscriber(
        api_key=key,
        prompt="A grocery shopping question for BigBasket",
    ) as client:
        return client.transcribe_turn(pcm)


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    products = 0
    load_error: Optional[str] = None
    try:
        products = len(get_pipeline().ids)
    except Exception as exc:  # surface index problems in the UI instead of hiding them
        load_error = str(exc)

    ready = bool(voice_key()) and products > 0
    message = ""
    if not voice_key():
        message = "Add OPENAI_API to .env to enable voice transcription."
    elif load_error:
        message = f"Index unavailable: {load_error}"

    return {
        "status": "ok",
        "voice_ready": ready,
        "index_ready": products > 0,
        "products": products,
        "message": message,
    }


@app.post("/api/voice/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> Dict[str, str]:
    blob = await audio.read()
    if not blob:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        pcm = wav_to_pcm16_24k(blob)
    except TranscriptionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ~200 ms of 24 kHz 16-bit mono; shorter clips are almost always clicks.
    if len(pcm) < 9600:
        raise HTTPException(status_code=400, detail="Recording is too short")

    key = voice_key()
    if not key:
        raise HTTPException(status_code=503, detail="Voice transcription is not configured")

    try:
        transcript = await run_in_threadpool(_transcribe, pcm, key)
    except (TranscriptionError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"transcript": transcript.strip()}


@app.post("/api/ask")
async def ask(body: AskBody) -> Dict[str, Any]:
    result = await run_in_threadpool(
        get_pipeline().answer,
        body.question,
        k=body.k,
        mode=body.mode,
        chat_history=body.history or None,
        **live_request_kwargs(
            body.live, body.platform, body.lat, body.lon, body.pincode
        ),
    )
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "confidence": result.get("confidence_score"),
        "latency_ms": result.get("latency_ms"),
        "cached": result.get("cached", False),
        "fallback": result.get("fallback", False),
    }
