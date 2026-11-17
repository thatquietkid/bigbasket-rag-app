/**
 * MediaRecorder emits webm/ogg (Chrome) or mp4 (Safari). stt.py expects a WAV,
 * so decode in the browser and write a 16-bit mono WAV before uploading.
 */

function writeAscii(view: DataView, offset: number, value: string) {
  for (let i = 0; i < value.length; i++) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

export function encodeWav(buffer: AudioBuffer): Blob {
  const frames = buffer.length;
  const dataBytes = frames * 2; // mono, 16-bit
  const bytes = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(bytes);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * 2, true);
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeAscii(view, 36, "data");
  view.setUint32(40, dataBytes, true);

  const channels = buffer.numberOfChannels;
  const data = Array.from({ length: channels }, (_, i) => buffer.getChannelData(i));
  let offset = 44;
  for (let i = 0; i < frames; i++) {
    let sample = 0;
    for (let c = 0; c < channels; c++) sample += data[c][i];
    sample /= channels;
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([bytes], { type: "audio/wav" });
}

export async function toWav(blob: Blob): Promise<Blob> {
  const ctx = new AudioContext();
  try {
    const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
    return encodeWav(decoded);
  } finally {
    if (ctx.state !== "closed") await ctx.close();
  }
}
