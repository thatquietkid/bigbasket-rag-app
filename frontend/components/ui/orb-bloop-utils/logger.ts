type Level = "debug" | "info" | "warn" | "error";

const enabled: Record<Level, boolean> = {
  debug: Boolean(import.meta.env.DEV),
  info: true,
  warn: true,
  error: true,
};

function emit(level: Level, scope: string, args: unknown[]) {
  if (!enabled[level]) return;
  const prefix = `[OrbBloop:${scope}]`;
  const method = level === "debug" ? "log" : level;
  // eslint-disable-next-line no-console
  (console[method] as (...values: unknown[]) => void)(prefix, ...args);
}

export const logger = {
  debug: (...args: unknown[]) => emit("debug", "debug", args),
  info: (...args: unknown[]) => emit("info", "info", args),
  warn: (...args: unknown[]) => emit("warn", "warn", args),
  error: (...args: unknown[]) => emit("error", "error", args),
};
