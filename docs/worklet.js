/* Audio worklet: hands raw mono blocks to the page and nothing else.
   Capturing uncompressed samples means the measurement sees exactly what the microphone produced,
   with no codec in between and no MediaRecorder container differences between browsers. */
class TapProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) this.port.postMessage(new Float32Array(ch));
    return true;
  }
}
registerProcessor("tap", TapProcessor);
