import { logger } from "./logger";

export interface GpuGateState {
  /** Reduce resolution and effects on small/low-core/reduced-motion devices. */
  lowPower: boolean;
  /** Pause the render loop while off-screen or in a background tab. */
  paused: boolean;
  /** Minimum milliseconds between draws. 0 = render every frame. */
  frameMs: number;
}

export interface GpuGate {
  state: GpuGateState;
  /** Mirror of state.frameMs for the supplied orb component. */
  frameMs: number;
  dispose(): void;
}

/**
 * Keeps WebGPU work proportional to what the user can actually see.
 * The orb reads `state` every frame, so the gate never needs to re-create
 * the GPU pipeline.
 */
export function attachGpuGate(canvas: HTMLCanvasElement): GpuGate {
  const reducedMotion =
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const lowPower =
    reducedMotion ||
    (typeof navigator !== "undefined" && (navigator.hardwareConcurrency || 8) <= 4) ||
    (typeof window !== "undefined" && window.innerWidth < 640);

  const state: GpuGateState = {
    lowPower,
    paused: typeof document !== "undefined" && document.hidden,
    frameMs: lowPower ? 1000 / 30 : 0,
  };

  let onScreen = true;
  let tabVisible = typeof document === "undefined" || !document.hidden;
  const sync = () => {
    state.paused = !onScreen || !tabVisible;
  };

  let observer: IntersectionObserver | undefined;
  if (typeof IntersectionObserver !== "undefined") {
    observer = new IntersectionObserver(
      (entries) => {
        onScreen = entries.some((entry) => entry.isIntersecting);
        sync();
      },
      { threshold: 0.01 },
    );
    observer.observe(canvas);
  }

  const onVisibility = () => {
    tabVisible = !document.hidden;
    sync();
  };
  if (typeof document !== "undefined") {
    document.addEventListener("visibilitychange", onVisibility);
  }

  sync();
  logger.debug("GPU gate attached", { ...state });

  return {
    state,
    frameMs: state.frameMs,
    dispose() {
      observer?.disconnect();
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", onVisibility);
      }
    },
  };
}

/**
 * Defers the expensive WebGPU init until the canvas is near the viewport.
 * Returns a disposer so React strict mode and unmounts cannot leak observers.
 */
export function deferUntilVisible(
  element: Element,
  run: () => void,
): () => void {
  if (typeof IntersectionObserver === "undefined") {
    const id = requestAnimationFrame(run);
    return () => cancelAnimationFrame(id);
  }

  let fired = false;
  const observer = new IntersectionObserver(
    (entries) => {
      if (fired || !entries.some((entry) => entry.isIntersecting)) return;
      fired = true;
      observer.disconnect();
      run();
    },
    { rootMargin: "160px" },
  );
  observer.observe(element);

  return () => {
    fired = true;
    observer.disconnect();
  };
}
