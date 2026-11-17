import type { RGB } from "./types";

export enum BloopPaletteName {
  blue = "blue",
  violet = "violet",
  mint = "mint",
  amber = "amber",
  rose = "rose",
  ocean = "ocean",
}

export interface BloopPalette {
  main: RGB;
  low: RGB;
  mid: RGB;
  high: RGB;
}

/** Values are linear-ish 0–1 RGB, matching the orb's shader inputs. */
export const BLOOP_PALETTES: Record<BloopPaletteName, BloopPalette> = {
  [BloopPaletteName.blue]: {
    main: [0.1, 0.5, 1.0],
    low: [0.1, 0.2, 0.8],
    mid: [0.2, 0.4, 0.9],
    high: [0.5, 0.8, 1.0],
  },
  [BloopPaletteName.violet]: {
    main: [0.42, 0.39, 1.0],
    low: [0.24, 0.18, 0.78],
    mid: [0.38, 0.32, 0.92],
    high: [0.72, 0.66, 1.0],
  },
  [BloopPaletteName.mint]: {
    main: [0.12, 0.72, 0.62],
    low: [0.06, 0.38, 0.44],
    mid: [0.1, 0.58, 0.58],
    high: [0.55, 0.95, 0.82],
  },
  [BloopPaletteName.amber]: {
    main: [0.98, 0.62, 0.16],
    low: [0.62, 0.28, 0.08],
    mid: [0.86, 0.48, 0.14],
    high: [1.0, 0.86, 0.55],
  },
  [BloopPaletteName.rose]: {
    main: [0.94, 0.36, 0.55],
    low: [0.58, 0.14, 0.38],
    mid: [0.8, 0.28, 0.5],
    high: [1.0, 0.72, 0.82],
  },
  [BloopPaletteName.ocean]: {
    main: [0.12, 0.66, 0.86],
    low: [0.06, 0.26, 0.6],
    mid: [0.1, 0.46, 0.78],
    high: [0.6, 0.92, 1.0],
  },
};
