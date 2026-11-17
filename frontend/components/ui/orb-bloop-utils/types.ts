import type { CSSProperties, RefObject } from "react";

/** Visual state of the orb. Drives colour, motion and pulse in the shader. */
export enum BloopState {
  idle = "idle",
  listen = "listen",
  think = "think",
  speak = "speak",
}

export type RGB = [number, number, number];
export type RGBA = [number, number, number, number];

export type AudioMode = "ambient" | "mic" | "file";

/** Analyser shape consumed by orb-bloop.tsx. Values are 0–255 byte averages. */
export interface AudioAnalyzer {
  update(): void;
  allAvg: number;
  lowAvg: number;
  midAvg: number;
  highAvg: number;
}

export type AudioElementRef = RefObject<HTMLAudioElement | null>;

export interface OrbBloopProps {
  audioMode?: AudioMode;
  audioElement?: AudioElementRef;
  audioSrc?: string;
  state?: BloopState;
  bloopColorMain?: RGB;
  bloopColorLow?: RGB;
  bloopColorMid?: RGB;
  bloopColorHigh?: RGB;
  size?: number;
  watercolorStrength?: number;
  watercolorAnimated?: boolean;
  className?: string;
  style?: CSSProperties;
}
