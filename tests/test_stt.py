"""Tests for stt.RealtimeTranscriber (OpenAI Realtime transcription) and WAV conversion."""

import base64
import io
import json
import math
import struct
import wave

import pytest

from stt import CHUNK_BYTES, RealtimeTranscriber, TranscriptionError, wav_to_pcm16_24k


class FakeConn:
    """Records what the transcriber sends and replays queued server events."""

    def __init__(self, events=None):
        self.sent = []
        self.events = list(events or [])
        self.closed = False

    def send(self, data):
        self.sent.append(json.loads(data))

    def recv(self, timeout=None):
        if not self.events:
            raise TimeoutError("no more events")
        return json.dumps(self.events.pop(0))

    def close(self):
        self.closed = True


def make_transcriber(events=None, **kwargs):
    conn = FakeConn(events)
    seen = {}

    def connector(url, headers):
        seen["url"] = url
        seen["headers"] = headers
        return conn

    transcriber = RealtimeTranscriber(api_key="test-key", connector=connector, **kwargs)
    return transcriber, conn, seen


def completed(transcript, item_id="item_003"):
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
    }


def delta(text, item_id="item_003"):
    return {
        "type": "conversation.item.input_audio_transcription.delta",
        "item_id": item_id,
        "content_index": 0,
        "delta": text,
    }


def test_open_connects_to_the_transcription_intent_with_the_api_key():
    transcriber, conn, seen = make_transcriber()

    transcriber.open()

    assert seen["url"] == "wss://api.openai.com/v1/realtime?intent=transcription"
    assert seen["headers"]["Authorization"] == "Bearer test-key"


def test_connection_sends_only_bearer_auth():
    # The live endpoint rejects the legacy OpenAI-Beta: realtime=v1 header with
    # invalid_request_error.beta_api_shape_disabled.
    transcriber, conn, seen = make_transcriber()

    transcriber.open()

    assert seen["headers"] == {"Authorization": "Bearer test-key"}


def test_open_sends_a_live_transcribe_session_update():
    transcriber, conn, _ = make_transcriber()

    transcriber.open()

    assert conn.sent[0] == {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "gpt-live-transcribe"},
                    "turn_detection": None,
                }
            },
        },
    }


def test_session_update_carries_prompt_keywords_languages_and_delay():
    transcriber, conn, _ = make_transcriber(
        prompt="A grocery support call",
        keywords=["Amul", "atta"],
        languages=["en", "hi"],
        delay="low",
    )

    transcriber.open()

    transcription = conn.sent[0]["session"]["audio"]["input"]["transcription"]
    assert transcription["prompt"] == "A grocery support call"
    assert transcription["keywords"] == ["Amul", "atta"]
    assert transcription["languages"] == ["en", "hi"]
    assert transcription["delay"] == "low"
    assert "language" not in transcription


def test_turn_streams_audio_in_chunks_then_commits():
    pcm = bytes(range(256)) * 80  # 20480 bytes
    transcriber, conn, _ = make_transcriber(events=[completed("hello")])
    transcriber.open()

    transcriber.transcribe_turn(pcm)

    appends = [m for m in conn.sent if m.get("type") == "input_audio_buffer.append"]
    assert [len(base64.b64decode(a["audio"])) for a in appends] == [CHUNK_BYTES, CHUNK_BYTES, 20480 - 2 * CHUNK_BYTES]
    assert b"".join(base64.b64decode(a["audio"]) for a in appends) == pcm
    assert conn.sent[-1]["type"] == "input_audio_buffer.commit"


def test_turn_returns_final_transcript_and_forwards_deltas():
    deltas = []
    transcriber, conn, _ = make_transcriber(
        events=[delta("Hello,"), delta(" how are you?"), completed("Hello, how are you?")],
        on_delta=deltas.append,
    )
    transcriber.open()

    transcript = transcriber.transcribe_turn(b"\x00\x00" * 480)

    assert transcript == "Hello, how are you?"
    assert deltas == ["Hello,", " how are you?"]


def test_turn_matches_completion_to_its_committed_item():
    events = [
        {"type": "input_audio_buffer.committed", "item_id": "item_001"},
        delta("Bonjour", item_id="item_002"),
        completed("Other turn", item_id="item_002"),
        completed("Bonjour", item_id="item_001"),
    ]
    transcriber, conn, _ = make_transcriber(events=events)
    transcriber.open()

    assert transcriber.transcribe_turn(b"\x00\x00" * 480) == "Bonjour"


def test_failed_transcription_event_raises():
    transcriber, conn, _ = make_transcriber(
        events=[
            {"type": "input_audio_buffer.committed", "item_id": "item_009"},
            {
                "type": "conversation.item.input_audio_transcription.failed",
                "item_id": "item_009",
                "error": {"message": "audio too quiet"},
            },
        ]
    )
    transcriber.open()

    with pytest.raises(TranscriptionError, match="too quiet"):
        transcriber.transcribe_turn(b"\x00\x00" * 480)


def test_realtime_error_event_raises():
    transcriber, conn, _ = make_transcriber(
        events=[{"type": "error", "error": {"type": "invalid_request", "message": "bad session"}}]
    )
    transcriber.open()

    with pytest.raises(TranscriptionError, match="bad session"):
        transcriber.transcribe_turn(b"\x00\x00" * 480)


def test_api_key_is_read_from_the_OPENAI_API_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API", "env-openai-key")
    assert RealtimeTranscriber().api_key == "env-openai-key"


def test_missing_api_key_raises_value_error_naming_the_env_var(monkeypatch):
    monkeypatch.delenv("OPENAI_API", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API"):
        RealtimeTranscriber()


def test_context_manager_closes_the_connection():
    transcriber, conn, _ = make_transcriber()

    with transcriber:
        pass

    assert conn.closed is True


def build_wav(seconds=1.0, rate=8000, channels=1, freq=440.0):
    frames = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        samples = []
        for i in range(frames):
            value = int(12000 * math.sin(2 * math.pi * freq * i / rate))
            samples.extend([value] * channels)
        w.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return buf.getvalue()


def test_wav_is_resampled_to_24k_mono_pcm16():
    pcm = wav_to_pcm16_24k(build_wav(seconds=1.0, rate=8000))

    assert isinstance(pcm, bytes)
    # 1 second at 24 kHz, 16-bit mono = 48000 bytes
    assert 47000 <= len(pcm) <= 48500


def test_stereo_wav_is_downmixed_to_mono():
    pcm = wav_to_pcm16_24k(build_wav(seconds=1.0, rate=24000, channels=2))

    assert 47000 <= len(pcm) <= 48500
