"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import {
  attachGpuGate,
  deferUntilVisible,
} from "@/components/ui/orb-bloop-utils/gpu-runtime";
import { logger } from "@/components/ui/orb-bloop-utils/logger";
import {
  BloopState,
  type OrbBloopProps,
} from "@/components/ui/orb-bloop-utils/types";
import { useOrbAudio } from "@/components/ui/orb-bloop-utils/use-orb-audio";
import { BLOOP_WGSL } from "@/components/ui/orb-bloop-utils/bloop-wgsl";

export function OrbBloop({
  audioMode = "ambient",
  audioElement,
  audioSrc,
  state = BloopState.idle,
  bloopColorMain = [0.1, 0.5, 1.0],
  bloopColorLow = [0.1, 0.2, 0.8],
  bloopColorMid = [0.2, 0.4, 0.9],
  bloopColorHigh = [0.5, 0.8, 1.0],
  size = 144,
  watercolorStrength = 0.5,
  watercolorAnimated = false,
  className = "",
  style,
}: OrbBloopProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioAnalyzerRef = useOrbAudio(audioMode, audioElement, audioSrc);
  const startTime = useRef(Date.now());
  const stateTrackingRef = useRef({ current: state, enteredAt: 0 });
  const audioAverageRef = useRef([0, 0, 0, 0]);
  const cumulativeAudioRef = useRef([0, 0, 0, 0]);
  const lastTimeRef = useRef(Date.now());
  const propsRef = useRef({
    state,
    audioMode,
    bloopColorMain,
    bloopColorLow,
    bloopColorMid,
    bloopColorHigh,
    watercolorStrength,
    watercolorAnimated,
  });
  propsRef.current = {
    state,
    audioMode,
    bloopColorMain,
    bloopColorLow,
    bloopColorMid,
    bloopColorHigh,
    watercolorStrength,
    watercolorAnimated,
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    let stop: (() => void) | undefined;
    let disposeGpu: (() => void) | undefined;
    let disposeGate: (() => void) | undefined;
    let disposeDefer: (() => void) | undefined;

    // Browsers without a WebGPU adapter still get an animated, state-aware orb.
    function startFallback(target: HTMLCanvasElement) {
      const context = target.getContext("2d");
      if (!context) return () => {};
      let frameId = 0;

      const draw = () => {
        const now = Date.now();
        const time = (now - startTime.current) / 1000;
        const p = propsRef.current;
        const analyzer = audioAnalyzerRef.current;
        let mic = 0;
        if (analyzer) {
          analyzer.update();
          mic = Math.min(1, analyzer.allAvg / 255);
        } else if (p.audioMode === "ambient") {
          mic = Math.sin(time * 2) * 0.1 + 0.12;
        }

        if (stateTrackingRef.current.current !== p.state) {
          stateTrackingRef.current.current = p.state;
          stateTrackingRef.current.enteredAt = time;
        }
        const age = time - stateTrackingRef.current.enteredAt;
        const listening = p.state === BloopState.listen;
        const thinking = p.state === BloopState.think;
        const speaking = p.state === BloopState.speak;
        const pulse = listening ? Math.sin(age * 7) * 0.05 : 0;
        const wobble =
          Math.sin(time * 1.7) * 0.02 +
          Math.sin(time * 3.3 + 1.4) * 0.012 +
          mic * 0.09 +
          pulse;

        const w = target.width;
        const h = target.height;
        const cx = w / 2;
        const cy = h / 2;
        const base = Math.min(w, h) * 0.36 * (1 + wobble);
        context.clearRect(0, 0, w, h);

        if (thinking) {
          context.save();
          context.translate(cx, cy);
          context.rotate(age * 1.6);
          for (let i = 0; i < 3; i++) {
            context.beginPath();
            context.arc(0, 0, base * (1.12 + i * 0.14), 0, Math.PI * 1.4);
            context.strokeStyle = `rgba(${p.bloopColorMid.map((c) => Math.round(c * 255)).join(",")},${0.3 - i * 0.08})`;
            context.lineWidth = w * 0.012;
            context.stroke();
          }
          context.restore();
        }

        if (speaking) {
          for (let i = 0; i < 2; i++) {
            const phase = (age * 0.8 + i * 0.5) % 1;
            context.beginPath();
            context.arc(cx, cy, base * (1 + phase * 0.7), 0, Math.PI * 2);
            context.strokeStyle = `rgba(${p.bloopColorHigh.map((c) => Math.round(c * 255)).join(",")},${0.35 * (1 - phase)})`;
            context.lineWidth = w * 0.014;
            context.stroke();
          }
        }

        const gradient = context.createRadialGradient(
          cx - base * 0.25,
          cy - base * 0.3,
          base * 0.08,
          cx,
          cy,
          base,
        );
        const rgb = (color: readonly number[]) =>
          color.map((c) => Math.round(Math.max(0, Math.min(1, c)) * 255)).join(",");
        gradient.addColorStop(0, `rgba(${rgb(p.bloopColorHigh)},0.98)`);
        gradient.addColorStop(0.42, `rgba(${rgb(p.bloopColorMain)},0.94)`);
        gradient.addColorStop(0.82, `rgba(${rgb(p.bloopColorMid)},0.85)`);
        gradient.addColorStop(1, `rgba(${rgb(p.bloopColorLow)},0)`);

        context.beginPath();
        context.arc(cx, cy, base, 0, Math.PI * 2);
        context.fillStyle = gradient;
        context.fill();
        frameId = requestAnimationFrame(draw);
      };

      draw();
      return () => cancelAnimationFrame(frameId);
    }

    async function mount() {
      const { init, effect, surface, frameLoop } = await import("vgpu");
      const activeCanvas = canvasRef.current;
      if (cancelled || !activeCanvas) return;
      const gpu = await init();
      if (cancelled) {
        gpu.dispose();
        return;
      }

      const gate = attachGpuGate(activeCanvas);
      disposeGate = gate.dispose;
      if (gate.state.lowPower) {
        activeCanvas.width = 160;
        activeCanvas.height = 160;
      }
      const canvasSurface = surface(gpu, activeCanvas, {
        format: "bgra8unorm",
        alphaMode: "premultiplied",
        ...(gate.state.lowPower ? { dpr: 1 } : {}),
      });
      const fx = effect(gpu, BLOOP_WGSL, { blend: "premultiplied" });
      let lastDraw = 0;
      const loop = frameLoop(gpu, (frame) => {
        if (gate.state.paused) return;
        const now = Date.now();
        if (gate.frameMs && now - lastDraw < gate.frameMs) return;
        lastDraw = now;
        const time = (now - startTime.current) / 1000;
        let avg = [0, 0, 0, 0];
        let micLevel = 0;
        if (audioAnalyzerRef.current) {
          audioAnalyzerRef.current.update();
          const a = audioAnalyzerRef.current;
          avg = [
            a.allAvg / 255,
            a.lowAvg / 255,
            a.midAvg / 255,
            a.highAvg / 255,
          ];
          micLevel = avg[0];
        } else if (propsRef.current.audioMode === "ambient") {
          micLevel = Math.sin(time * 2.0) * 0.1 + 0.1;
          avg = [micLevel, micLevel * 0.6, micLevel * 0.4, micLevel];
        }
        const dt = Math.min(now - lastTimeRef.current, 100) / 1000;
        lastTimeRef.current = now;
        for (let i = 0; i < 4; i++) {
          audioAverageRef.current[i] +=
            (avg[i] - audioAverageRef.current[i]) * 0.55;
          cumulativeAudioRef.current[i] +=
            audioAverageRef.current[i] * (60 * dt) * 0.25;
        }
        const p = propsRef.current;
        const activeState = p.state;
        if (stateTrackingRef.current.current !== activeState) {
          stateTrackingRef.current.current = activeState;
          stateTrackingRef.current.enteredAt = time;
        }
        const enteredAt = stateTrackingRef.current.enteredAt;

        fx.set({
          ubo: {
            time,
            micLevel,
            stateListen: activeState === BloopState.listen ? 1 : 0,
            listenTimestamp: activeState === BloopState.listen ? enteredAt : 0,
            stateThink: activeState === BloopState.think ? 1 : 0,
            thinkTimestamp: activeState === BloopState.think ? enteredAt : 0,
            stateSpeak: activeState === BloopState.speak ? 1 : 0,
            speakTimestamp: activeState === BloopState.speak ? enteredAt : 0,
            avgMag: audioAverageRef.current,
            cumulativeAudio: cumulativeAudioRef.current,
            viewport: [
              canvasRef.current?.width || size,
              canvasRef.current?.height || size,
            ],
            watercolorStrength: gate.state.lowPower ? 0 : p.watercolorStrength,
            watercolorAnimated: 0,
            bloopColorMain: [...p.bloopColorMain, 1],
            bloopColorLow: [...p.bloopColorLow, 1],
            bloopColorMid: [...p.bloopColorMid, 1],
            bloopColorHigh: [...p.bloopColorHigh, 1],
          },
        });
        frame.pass({ target: canvasSurface, clear: [0, 0, 0, 0] }, fx);
      });
      stop = () => loop.stop();
      disposeGpu = () => gpu.dispose();
    }

    disposeDefer = deferUntilVisible(canvas, () => {
      mount().catch((error) => {
        logger.warn("WebGPU unavailable, using 2D orb fallback:", error);
        stop = startFallback(canvas);
      });
    });
    return () => {
      cancelled = true;
      disposeDefer?.();
      stop?.();
      disposeGate?.();
      disposeGpu?.();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={512}
      height={512}
      className={cn("aspect-square", className)}
      style={{
        width: size !== undefined ? `${size}px` : "100%",
        height: size !== undefined ? `${size}px` : "100%",
        ...style,
      }}
    />
  );
}

export default OrbBloop;
