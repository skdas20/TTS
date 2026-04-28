// AudioWorklet: downsample mic input to 16 kHz mono PCM16 and post to main thread.
class DownsamplerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.sourceSampleRate = sampleRate;
    this.step = this.sourceSampleRate / this.targetSampleRate;
    this.position = 0;
    this.history = new Float32Array(0);
    this.output = [];
    this.flushSize = 1024;
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input || input.length === 0) return true;

    const merged = new Float32Array(this.history.length + input.length);
    merged.set(this.history, 0);
    merged.set(input, this.history.length);

    while (this.position + 1 < merged.length) {
      const leftIndex = Math.floor(this.position);
      const rightIndex = leftIndex + 1;
      const frac = this.position - leftIndex;
      const left = merged[leftIndex];
      const right = merged[rightIndex];
      const sample = left + (right - left) * frac;
      const clamped = Math.max(-1, Math.min(1, sample));
      this.output.push(Math.round(clamped * 32767));
      this.position += this.step;
    }

    const consumed = Math.floor(this.position);
    this.history = merged.slice(consumed);
    this.position -= consumed;

    if (this.output.length >= this.flushSize) {
      const chunk = new Int16Array(this.output);
      this.output = [];
      this.port.postMessage(chunk.buffer, [chunk.buffer]);
    }

    return true;
  }
}

registerProcessor("downsampler", DownsamplerProcessor);
