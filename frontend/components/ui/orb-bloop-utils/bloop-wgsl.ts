import { prepareShader } from "@vgpu/wgsl/prepare";

/**
 * Fullscreen WGSL effect for the orb.
 *
 * vgpu reflects the uniform binding named `ubo`, so the field names below must
 * stay in sync with the object passed to `fx.set({ ubo: { ... } })` in
 * orb-bloop.tsx. The JS arrays are packed as vec4/vec2, which keeps the
 * uniform layout free of array-stride surprises.
 *
 * `effect()` expects a reflected ShaderSource, so the string is prepared with
 * vgpu's own compiler rather than passed raw.
 */
export const BLOOP_WGSL = prepareShader(/* wgsl */ `
struct Ubo {
  time: f32,
  micLevel: f32,
  stateListen: f32,
  listenTimestamp: f32,
  stateThink: f32,
  thinkTimestamp: f32,
  stateSpeak: f32,
  speakTimestamp: f32,
  avgMag: vec4f,
  cumulativeAudio: vec4f,
  viewport: vec2f,
  watercolorStrength: f32,
  watercolorAnimated: f32,
  bloopColorMain: vec4f,
  bloopColorLow: vec4f,
  bloopColorMid: vec4f,
  bloopColorHigh: vec4f,
};

@group(0) @binding(0) var<uniform> ubo: Ubo;

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += vec2f(dot(q, q + vec2f(45.32)));
  return fract(q.x * q.y);
}

fn noise2(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let p = (uv - 0.5) * 2.0;
  let t = ubo.time;
  let radius = length(p);
  let angle = atan2(p.y, p.x);

  let mic = clamp(ubo.micLevel, 0.0, 1.0);
  let n1 = noise2(p * 3.2 + vec2f(t * 0.25, -t * 0.18));
  let n2 = noise2(p * 7.5 - vec2f(t * 0.4, t * 0.3));

  var distort = (n1 - 0.5) * 0.06 + (n2 - 0.5) * 0.025;
  distort += (ubo.avgMag.x - 0.1) * 0.06 * sin(angle * 5.0 + t * 2.0);
  distort += sin(t * (1.6 + mic * 4.0)) * 0.012 + mic * 0.02;

  let listenT = max(0.0, t - ubo.listenTimestamp);
  let thinkT = max(0.0, t - ubo.thinkTimestamp);
  let speakT = max(0.0, t - ubo.speakTimestamp);
  let ripple = sin(radius * 26.0 - t * 6.0) * 0.5 + 0.5;

  let listenBoost = ubo.stateListen * (0.018 * sin(listenT * 7.0) + mic * 0.05);
  let thinkWobble = ubo.stateThink * 0.03 * sin(angle * 3.0 + thinkT * 5.0) * exp(-thinkT * 0.6);
  let speakWave = ubo.stateSpeak * 0.035 * ripple * (0.5 + 0.5 * sin(speakT * 4.0));

  let edge = 0.72 + distort + listenBoost + thinkWobble + speakWave;
  let d = radius / max(edge, 0.05);
  let aa = fwidth(d) * 1.5 + 0.012;

  let body = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, d);
  let core = 1.0 - smoothstep(0.0, 0.86, d);
  let rim = smoothstep(0.62, 0.98, d) * body;

  var col = mix(ubo.bloopColorLow.rgb, ubo.bloopColorMain.rgb, core);
  col = mix(col, ubo.bloopColorHigh.rgb, rim * (0.35 + 0.5 * mic));
  col = mix(
    col,
    ubo.bloopColorMid.rgb,
    ubo.stateThink * 0.5 * (0.5 + 0.5 * sin(angle * 4.0 - t * 2.0))
  );
  col += ubo.bloopColorHigh.rgb * ubo.stateSpeak * ripple * 0.18;

  let swirl = noise2(p * 5.0 + vec2f(t * 0.15, t * 0.1));
  let breath = 1.0 + clamp(ubo.cumulativeAudio.x, 0.0, 1.0) * 0.2;
  col *= (0.82 + 0.28 * swirl) * breath;

  let grainSpeed = mix(2.0, 8.0, ubo.watercolorAnimated);
  let grain = hash21(uv * ubo.viewport + vec2f(t * grainSpeed, 0.0)) - 0.5;
  col += grain * ubo.watercolorStrength * 0.08;

  let alpha = clamp(body, 0.0, 1.0) * 0.96;
  // The effect is created with blend: "premultiplied".
  return vec4f(max(col, vec3f(0.0)) * alpha, alpha);
}
`, "bloop.wgsl");
