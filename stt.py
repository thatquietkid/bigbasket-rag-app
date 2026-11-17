"""
stt.py — Live speech-to-text via the OpenAI Realtime transcription API.

Streams mono 16-bit PCM at 24 kHz to
``wss://api.openai.com/v1/realtime?intent=transcription`` with model
``gpt-live-transcribe``, receives transcript deltas as speech arrives, and
returns the final transcript once the turn is committed.

Usage:
    with RealtimeTranscriber(on_delta=print) as stt:
        text = stt.transcribe_turn(wav_to_pcm16_24k(wav_bytes))

Key is read from the OPENAI_API environment variable (then OPENAI_API_KEY).
"""

import base64
import io
import json
import os
import time
import wave
from typing import Any, Callable, Dict, List, Optional

import numpy as np

OPENAI_API_ENV = "OPENAI_API"
REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
MODEL = "gpt-live-transcribe"
AUDIO_RATE = 24000
# 9600 bytes = 100 ms of 24 kHz 16-bit mono, the chunk size from OpenAI's example.
CHUNK_BYTES = 9600


class TranscriptionError(Exception):
    """Raised when the Realtime API cannot produce a transcript."""


def wav_to_pcm16_24k(wav_bytes: bytes, target_rate: int = AUDIO_RATE) -> bytes:
    """Convert a WAV recording into mono 16-bit PCM at 24 kHz for streaming."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            rate = w.getframerate()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        raise TranscriptionError(f"Unreadable audio: {e}") from e

    if width != 2:
        raise TranscriptionError(f"Unsupported audio: {width * 8}-bit samples, need 16-bit PCM")

    data = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if rate != target_rate and data.size:
        duration = data.size / float(rate)
        target_n = max(1, int(round(duration * target_rate)))
        old_x = np.linspace(0.0, duration, num=data.size, endpoint=False)
        new_x = np.linspace(0.0, duration, num=target_n, endpoint=False)
        data = np.interp(new_x, old_x, data)

    return np.clip(np.round(data), -32768, 32767).astype("<i2").tobytes()


class RealtimeTranscriber:
    """One Realtime transcription session, reusable across audio turns."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = MODEL,
        url: str = REALTIME_URL,
        prompt: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        delay: Optional[str] = None,
        on_delta: Optional[Callable[[str], None]] = None,
        timeout: float = 90.0,
        connector: Optional[Callable[[str, Dict[str, str]], Any]] = None,
    ):
        self.api_key = (
            api_key or os.getenv(OPENAI_API_ENV, "") or os.getenv("OPENAI_API_KEY", "")
        )
        if not self.api_key:
            raise ValueError(
                f"No OpenAI API key found. Set the {OPENAI_API_ENV} environment variable."
            )
        self.model = model
        self.url = url
        self.prompt = prompt
        self.keywords = keywords
        self.languages = languages
        self.delay = delay
        self.on_delta = on_delta
        self.timeout = timeout
        self._connector = connector or self._default_connector
        self._conn = None

    # -- connection --------------------------------------------------------
    def _default_connector(self, url: str, headers: Dict[str, str]) -> Any:
        from websockets.sync.client import connect

        return connect(url, additional_headers=headers, open_timeout=self.timeout, max_size=None)

    def _headers(self) -> Dict[str, str]:
        # Only bearer auth: the legacy OpenAI-Beta header is rejected now.
        return {"Authorization": f"Bearer {self.api_key}"}

    def session_update(self) -> Dict[str, Any]:
        transcription: Dict[str, Any] = {"model": self.model}
        if self.prompt:
            transcription["prompt"] = self.prompt
        if self.keywords:
            transcription["keywords"] = list(self.keywords)
        if self.languages:
            transcription["languages"] = list(self.languages)
        if self.delay:
            transcription["delay"] = self.delay
        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": AUDIO_RATE},
                        "transcription": transcription,
                        # gpt-live-transcribe does not support server/semantic VAD.
                        "turn_detection": None,
                    }
                },
            },
        }

    def open(self) -> "RealtimeTranscriber":
        self.close()
        self._conn = self._connector(self.url, self._headers())
        self._send(self.session_update())
        return self

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __enter__(self) -> "RealtimeTranscriber":
        return self.open() if self._conn is None else self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _send(self, payload: Dict[str, Any]) -> None:
        if self._conn is None:
            raise TranscriptionError("Transcription session is not open")
        self._conn.send(json.dumps(payload))

    # -- transcription -----------------------------------------------------
    def transcribe_turn(self, pcm: bytes) -> str:
        """Stream one audio turn, commit it, and return its final transcript."""
        if self._conn is None:
            self.open()

        for start in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[start : start + CHUNK_BYTES]
            self._send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )
        self._send({"type": "input_audio_buffer.commit"})

        item_id: Optional[str] = None
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TranscriptionError("Timed out waiting for the transcript")
            try:
                raw = self._conn.recv(timeout=remaining)
            except TimeoutError as e:
                raise TranscriptionError("Timed out waiting for the transcript") from e
            except Exception as e:  # websockets ConnectionClosed and friends
                raise TranscriptionError(f"Transcription connection failed: {e}") from e

            event = json.loads(raw)
            kind = event.get("type")

            if kind == "input_audio_buffer.committed":
                item_id = event.get("item_id")
            elif kind == "conversation.item.input_audio_transcription.delta":
                if item_id is None or event.get("item_id") == item_id:
                    if self.on_delta and event.get("delta"):
                        self.on_delta(event["delta"])
            elif kind == "conversation.item.input_audio_transcription.completed":
                if item_id is None or event.get("item_id") == item_id:
                    return event.get("transcript", "")
            elif kind == "conversation.item.input_audio_transcription.failed":
                if item_id is None or event.get("item_id") == item_id:
                    error = event.get("error") or {}
                    raise TranscriptionError(error.get("message") or "Transcription failed")
            elif kind == "error":
                error = event.get("error") or {}
                raise TranscriptionError(error.get("message") or "Realtime API error")
