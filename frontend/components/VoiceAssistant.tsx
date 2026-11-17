"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  CircleAlert,
  Loader2,
  Mic,
  ShoppingBasket,
  Sparkles,
  Square,
  Volume2,
  VolumeX,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import OrbBloop from "@/components/ui/orb-bloop";
import {
  BLOOP_PALETTES,
  BloopPaletteName,
} from "@/components/ui/orb-bloop-utils/palettes";
import { BloopState } from "@/components/ui/orb-bloop-utils/types";
import { toWav } from "@/components/wav";
import { cn } from "@/lib/utils";

interface Health {
  status: string;
  voice_ready: boolean;
  index_ready: boolean;
  products: number;
  message?: string;
}

interface AskResponse {
  answer: string;
  confidence?: number;
  latency_ms?: number;
  cached?: boolean;
  sources?: Array<{ product?: string; brand?: string; price?: number }>;
}

interface Turn {
  role: "user" | "assistant";
  text: string;
}

const COLORS = BLOOP_PALETTES[BloopPaletteName.violet];

const STATE_COPY: Record<BloopState, { label: string; hint: string }> = {
  [BloopState.idle]: {
    label: "Ready when you are",
    hint: "Tap the microphone and ask about products, prices or policies.",
  },
  [BloopState.listen]: {
    label: "Listening…",
    hint: "Speak naturally. Tap again when you're finished.",
  },
  [BloopState.think]: {
    label: "Thinking…",
    hint: "Transcribing your question and checking the catalogue.",
  },
  [BloopState.speak]: {
    label: "Answering…",
    hint: "Reading the verified answer out loud.",
  },
};

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string; error?: string };
    return body.detail || body.error || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

export function VoiceAssistant() {
  const [state, setState] = useState<BloopState>(BloopState.idle);
  const [hearing, setHearing] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [muted, setMuted] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const turnsRef = useRef<Turn[]>([]);
  const mutedRef = useRef(muted);

  useEffect(() => {
    turnsRef.current = turns;
  }, [turns]);
  useEffect(() => {
    mutedRef.current = muted;
  }, [muted]);

  useEffect(() => {
    let alive = true;
    fetch("/api/health")
      .then((r) => (r.ok ? (r.json() as Promise<Health>) : Promise.reject()))
      .then((h) => {
        if (alive) setHealth(h);
      })
      .catch(() => {
        if (alive) setHealth(null);
      });
    return () => {
      alive = false;
      window.speechSynthesis?.cancel();
      recorderRef.current?.state !== "inactive" && recorderRef.current?.stop();
    };
  }, []);

  const speak = useCallback((text: string, done: () => void) => {
    if (!("speechSynthesis" in window)) {
      done();
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.03;
    utterance.pitch = 1;
    utterance.onend = done;
    utterance.onerror = done;
    window.speechSynthesis.speak(utterance);
  }, []);

  const processAudio = useCallback(
    async (raw: Blob) => {
      setBusy(true);
      setState(BloopState.think);
      try {
        const wav = await toWav(raw);
        const form = new FormData();
        form.append("audio", wav, "question.wav");

        const transcription = await fetch("/api/voice/transcribe", {
          method: "POST",
          body: form,
        });
        if (!transcription.ok) throw new Error(await errorMessage(transcription));
        const { transcript } = (await transcription.json()) as {
          transcript: string;
        };
        const question = transcript.trim();
        if (!question) throw new Error("No speech detected — try recording again.");
        setHearing(question);

        const answerResponse = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            history: turnsRef.current.slice(-4),
          }),
        });
        if (!answerResponse.ok) throw new Error(await errorMessage(answerResponse));
        const answer = (await answerResponse.json()) as AskResponse;

        setTurns((previous) => [
          ...previous,
          { role: "user", text: question },
          { role: "assistant", text: answer.answer },
        ]);

        if (mutedRef.current || !answer.answer) {
          setState(BloopState.idle);
        } else {
          setState(BloopState.speak);
          speak(answer.answer, () => setState(BloopState.idle));
        }
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Something went wrong.");
        setState(BloopState.idle);
      } finally {
        setBusy(false);
      }
    },
    [speak],
  );

  const toggleRecording = useCallback(async () => {
    if (busy) return;
    const recorder = recorderRef.current;

    if (recorder && recorder.state === "recording") {
      recorder.stop();
      return;
    }

    setError("");
    setHearing("");
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Your browser does not support microphone access.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "";
      const next = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined,
      );
      chunksRef.current = [];
      next.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      next.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(chunksRef.current, {
          type: next.mimeType || "audio/webm",
        });
        if (!blob.size) {
          setError("No audio captured. Please try again.");
          setState(BloopState.idle);
          return;
        }
        void processAudio(blob);
      };
      recorderRef.current = next;
      next.start(200);
      setState(BloopState.listen);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Microphone access was denied.",
      );
      setState(BloopState.idle);
    }
  }, [busy, processAudio]);

  const toggleMute = () => {
    setMuted((previous) => {
      if (!previous) window.speechSynthesis?.cancel();
      return !previous;
    });
  };

  const listening = state === BloopState.listen;
  const copy = STATE_COPY[state];
  const backendReady = health?.status === "ok";

  return (
    <div className="relative min-h-dvh overflow-hidden bg-[var(--surface)]">
      {/* Decorative stock imagery, kept faint so the soft-UI surface leads. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 opacity-[0.07]"
        style={{
          backgroundImage:
            "url('https://images.unsplash.com/photo-1542838132-92c53300491e?auto=format&fit=crop&w=1920&q=70')",
          backgroundSize: "cover",
          backgroundPosition: "center",
        }}
      />

      <div className="relative mx-auto flex min-h-dvh w-full max-w-5xl flex-col gap-5 px-4 py-5 sm:gap-7 sm:px-8 sm:py-8">
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <span className="neo-raised-sm grid size-11 place-items-center rounded-2xl text-primary">
              <ShoppingBasket className="size-5" aria-hidden="true" />
            </span>
            <div>
              <p className="text-display text-base font-extrabold sm:text-lg">
                BigBasket <span className="text-primary">Voice</span>
              </p>
              <p className="text-xs text-muted-foreground">
                Catalogue-grounded answers, spoken aloud
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "neo-inset flex items-center gap-2 rounded-full px-3 py-2 text-xs font-semibold",
                backendReady ? "text-emerald-700" : "text-[var(--destructive)]",
              )}
            >
              <span
                className={cn(
                  "size-2 rounded-full",
                  backendReady ? "bg-emerald-600" : "bg-[var(--destructive)]",
                )}
              />
              {backendReady ? "API online" : "API offline"}
            </span>
            <Button variant="ghost" size="icon" asChild className="neo-raised-sm rounded-full">
              <a href="http://127.0.0.1:8501/" aria-label="Open the chat UI">
                <ArrowUpRight className="size-4" aria-hidden="true" />
              </a>
            </Button>
          </div>
        </header>

        <main className="grid flex-1 items-center gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] lg:gap-8">
          <section className="neo-raised flex flex-col items-center gap-5 rounded-[2rem] px-5 py-7 sm:px-8 sm:py-9">
            <div
              className="neo-well relative grid place-items-center rounded-full p-3"
              style={{ width: "clamp(180px, 30vw, 260px)" }}
            >
              <span className="block aspect-square w-full">
                <OrbBloop
                  state={state}
                  audioMode={listening ? "mic" : "ambient"}
                  size={undefined}
                  bloopColorMain={COLORS.main}
                  bloopColorLow={COLORS.low}
                  bloopColorMid={COLORS.mid}
                  bloopColorHigh={COLORS.high}
                  watercolorStrength={0.45}
                  style={{ width: "100%", height: "100%" }}
                  className="rounded-full"
                />
              </span>
              {listening ? (
                <span className="absolute inset-1 animate-ping rounded-full border-2 border-primary/30" />
              ) : null}
            </div>

            <div className="text-center">
              <p className="text-display flex items-center justify-center gap-2 text-xl font-extrabold sm:text-2xl">
                {busy ? (
                  <Loader2 className="size-5 animate-spin text-primary" aria-hidden="true" />
                ) : null}
                {copy.label}
              </p>
              <p className="mx-auto mt-1 max-w-xs text-sm text-muted-foreground">
                {copy.hint}
              </p>
            </div>

            <div className="flex items-center gap-3">
              <Button
                onClick={() => void toggleRecording()}
                disabled={busy || !backendReady}
                size="lg"
                className={cn(
                  "h-14 gap-2 rounded-2xl px-7 text-base font-bold shadow-[var(--raised-sm)] transition-all",
                  listening
                    ? "bg-[var(--destructive)] text-white hover:bg-[var(--destructive)]/90"
                    : "bg-primary text-primary-foreground hover:bg-primary/90",
                )}
              >
                {listening ? (
                  <Square className="size-5 fill-current" aria-hidden="true" />
                ) : (
                  <Mic className="size-5" aria-hidden="true" />
                )}
                {listening ? "Stop" : "Ask by voice"}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleMute}
                aria-label={muted ? "Unmute answers" : "Mute answers"}
                className="neo-raised-sm size-12 rounded-2xl"
              >
                {muted ? (
                  <VolumeX className="size-5" aria-hidden="true" />
                ) : (
                  <Volume2 className="size-5" aria-hidden="true" />
                )}
              </Button>
            </div>

            {error ? (
              <p
                role="alert"
                className="neo-inset flex items-center gap-2 rounded-2xl px-4 py-3 text-sm text-[var(--destructive)]"
              >
                <CircleAlert className="size-4 shrink-0" aria-hidden="true" />
                {error}
              </p>
            ) : null}
            {hearing && !error ? (
              <p className="neo-inset max-w-md rounded-2xl px-4 py-3 text-center text-sm">
                <span className="font-semibold text-primary">You said: </span>
                {hearing}
              </p>
            ) : null}
            {health?.message ? (
              <p className="text-center text-xs text-muted-foreground">
                {health.message}
              </p>
            ) : null}
          </section>

          <section className="flex min-h-[280px] flex-col gap-3 lg:max-h-[70vh]">
            <div className="flex items-center gap-2 px-1">
              <Sparkles className="size-4 text-primary" aria-hidden="true" />
              <h2 className="text-display text-sm font-bold uppercase tracking-widest text-muted-foreground">
                Conversation
              </h2>
            </div>

            <div className="neo-inset flex flex-1 flex-col gap-3 overflow-y-auto rounded-[2rem] p-4 sm:p-5">
              {turns.length === 0 ? (
                <div className="m-auto max-w-xs text-center text-sm text-muted-foreground">
                  Try “Which branded olive oils are under ₹500?”
                </div>
              ) : (
                turns.map((turn, index) => (
                  <div
                    key={`${turn.role}-${index}`}
                    className={cn(
                      "max-w-[92%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
                      turn.role === "user"
                        ? "neo-raised-sm self-end font-medium"
                        : "neo-well self-start",
                    )}
                  >
                    {turn.text}
                  </div>
                ))
              )}
            </div>

            {turns.length > 0 ? (
              <Button
                variant="ghost"
                onClick={() => {
                  setTurns([]);
                  setHearing("");
                  setError("");
                  setState(BloopState.idle);
                }}
                className="self-end rounded-xl text-xs text-muted-foreground"
              >
                Clear conversation
              </Button>
            ) : null}
          </section>
        </main>

        <footer className="flex flex-wrap items-center justify-between gap-2 px-1 text-xs text-muted-foreground">
          <span>
            {health
              ? `${health.products.toLocaleString()} products indexed · ${health.index_ready ? "retrieval ready" : "loading"}`
              : "Connect the voice API to begin"}
          </span>
          <span>Answers are grounded in the BigBasket catalogue.</span>
        </footer>
      </div>
    </div>
  );
}
