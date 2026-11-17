import { useEffect, useRef, useState } from "react";
import { logger } from "./logger";
import type {
  AudioAnalyzer,
  AudioElementRef,
  AudioMode,
} from "./types";

interface LiveAnalyzer {
  ctx: AudioContext;
  analyser: AnalyserNode;
  analyzer: AudioAnalyzer;
}

/** MediaElementSource may be created only once per <audio> element. */
const elementCache = new WeakMap<HTMLMediaElement, LiveAnalyzer>();

function buildAnalyzer(
  ctx: AudioContext,
  analyser: AnalyserNode,
  source: AudioNode,
): LiveAnalyzer {
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.78;
  source.connect(analyser);

  const bins = new Uint8Array(analyser.frequencyBinCount);
  const avgRange = (from: number, to: number) => {
    let sum = 0;
    for (let i = from; i < to; i++) sum += bins[i];
    return sum / Math.max(1, to - from);
  };

  const analyzer: AudioAnalyzer = {
    allAvg: 0,
    lowAvg: 0,
    midAvg: 0,
    highAvg: 0,
    update() {
      analyser.getByteFrequencyData(bins);
      const n = bins.length;
      const lowEnd = Math.max(1, Math.floor(n * 0.08));
      const midEnd = Math.floor(n * 0.35);
      analyzer.allAvg = avgRange(0, n);
      analyzer.lowAvg = avgRange(0, lowEnd);
      analyzer.midAvg = avgRange(lowEnd, midEnd);
      analyzer.highAvg = avgRange(midEnd, n);
    },
  };

  return { ctx, analyser, analyzer };
}

/** Returns a ref the orb polls each frame. Null means "use ambient motion". */
export function useOrbAudio(
  mode: AudioMode,
  audioElement?: AudioElementRef,
  audioSrc?: string,
) {
  const analyzerRef = useRef<AudioAnalyzer | null>(null);

  useEffect(() => {
    if (mode !== "mic" && mode !== "file") {
      analyzerRef.current = null;
      return;
    }

    let cancelled = false;
    let stream: MediaStream | undefined;
    let ownedCtx: AudioContext | undefined;
    let owned: LiveAnalyzer | undefined;

    const setup = async () => {
      try {
        if (mode === "mic") {
          if (!navigator.mediaDevices?.getUserMedia) {
            logger.warn("Microphone API is unavailable");
            return;
          }
          stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: false,
            },
          });
          if (cancelled) {
            stream.getTracks().forEach((track) => track.stop());
            return;
          }
          ownedCtx = new AudioContext();
          const analyser = ownedCtx.createAnalyser();
          owned = buildAnalyzer(
            ownedCtx,
            analyser,
            ownedCtx.createMediaStreamSource(stream),
          );
          analyzerRef.current = owned.analyzer;
          return;
        }

        // file mode
        const element = audioElement?.current;
        if (!element) return;

        const cached = elementCache.get(element);
        if (cached) {
          analyzerRef.current = cached.analyzer;
          return;
        }

        const ctx = new AudioContext();
        const analyser = ctx.createAnalyser();
        const live = buildAnalyzer(ctx, analyser, ctx.createMediaElementSource(element));
        elementCache.set(element, live);
        analyzerRef.current = live.analyzer;
      } catch (error) {
        logger.warn("Could not start orb audio analysis:", error);
      }
    };

    void setup();

    return () => {
      cancelled = true;
      analyzerRef.current = null;
      owned?.analyser.disconnect();
      stream?.getTracks().forEach((track) => track.stop());
      if (ownedCtx && ownedCtx.state !== "closed") {
        void ownedCtx.close();
      }
    };
  }, [mode, audioElement, audioSrc]);

  return analyzerRef;
}

/**
 * Converts a cross-origin audio URL into a same-origin blob URL so the Web
 * Audio analyser is not blocked by CORS. Object/data URLs pass through.
 */
export function useCorsAudioSrc(src: string): string {
  const [resolved, setResolved] = useState(src);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | undefined;

    setResolved(src);
    if (!src || src.startsWith("blob:") || src.startsWith("data:")) return;

    fetch(src, { mode: "cors" })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setResolved(objectUrl);
      })
      .catch((error) => {
        if (!cancelled) logger.warn("Audio CORS fetch failed, using original src", error);
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src]);

  return resolved;
}
