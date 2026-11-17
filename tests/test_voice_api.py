"""Tests for the FastAPI bridge used by the React voice orb."""

import io
import struct
import wave

import api
import pytest
from fastapi.testclient import TestClient


def wav_bytes(seconds: float = 1.0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(struct.pack("<h", 0) * int(24000 * seconds))
    return buffer.getvalue()


class FakeTranscriber:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def transcribe_turn(self, pcm):
        assert pcm
        return "  under ₹500 olive oil?  "


class FakePipeline:
    ids = ["a", "b"]

    def answer(self, question, **kwargs):
        return {
            "answer": "Saffola Gold is ₹499.",
            "sources": [{"product": "Saffola Gold"}],
            "confidence_score": 0.8,
            "latency_ms": 12.5,
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "get_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(api, "voice_key", lambda: "test-key")
    monkeypatch.setattr(api, "RealtimeTranscriber", FakeTranscriber)
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_reports_voice_and_index(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "voice_ready": True,
        "index_ready": True,
        "products": 2,
        "message": "",
    }


def test_transcribe_returns_trimmed_transcript(client):
    response = client.post(
        "/api/voice/transcribe",
        files={"audio": ("question.wav", wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json() == {"transcript": "under ₹500 olive oil?"}


def test_transcribe_rejects_short_audio(client):
    response = client.post(
        "/api/voice/transcribe",
        files={"audio": ("tiny.wav", wav_bytes(0.05), "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Recording is too short"


def test_ask_delegates_to_rag_pipeline(client):
    response = client.post(
        "/api/ask",
        json={"question": "Cheapest olive oil?", "history": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Saffola Gold is ₹499."
    assert body["confidence"] == 0.8
    assert body["sources"] == [{"product": "Saffola Gold"}]


def test_ask_validates_mode(client):
    response = client.post("/api/ask", json={"question": "hi", "mode": "wrong"})
    assert response.status_code == 422
