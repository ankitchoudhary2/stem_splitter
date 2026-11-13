// Ambience = the diffuse / noise-like residual left after removing the harmonic and percussive parts of "other".
import { stft, istft } from "./fft.js";
const SR = 44100, FFT = 4096, HOP = 1024, B = FFT / 2 + 1, W = 15, LIM = Math.log2(3);

function smoothTime(M, F, out) {
  const acc = new Float64Array(B);
  for (let f = 0; f <= Math.min(W, F - 1); f++) for (let b = 0; b < B; b++) acc[b] += M[f * B + b];
  for (let f = 0; f < F; f++) {
    const n = Math.min(F - 1, f + W) - Math.max(0, f - W) + 1;
    for (let b = 0; b < B; b++) out[f * B + b] = acc[b] / n;
    const add = f + W + 1, rem = f - W;
    if (add < F) for (let b = 0; b < B; b++) acc[b] += M[add * B + b];
    if (rem >= 0) for (let b = 0; b < B; b++) acc[b] -= M[rem * B + b];
  }
}
function smoothFreq(M, F, out) {
  for (let f = 0; f < F; f++) {
    const o = f * B; let acc = 0;
    for (let b = 0; b <= Math.min(W, B - 1); b++) acc += M[o + b];
    for (let b = 0; b < B; b++) {
      out[o + b] = acc / (Math.min(B - 1, b + W) - Math.max(0, b - W) + 1);
      const add = b + W + 1, rem = b - W;
      if (add < B) acc += M[o + add];
      if (rem >= 0) acc -= M[o + rem];
    }
  }
}
function residual(xl, xr) {
  const sl = stft(xl, FFT, HOP), sr = stft(xr, FFT, HOP), F = sl.numFrames, n = F * B;
  const M = new Float32Array(n), H = new Float32Array(n), P = new Float32Array(n);
  for (let i = 0; i < n; i++) M[i] = Math.hypot(sl.real[i], sl.imag[i], sr.real[i], sr.imag[i]);
  smoothTime(M, F, H); smoothFreq(M, F, P);
  for (let i = 0; i < n; i++) {           // mask is 1 where neither harmonic nor percussive clearly dominates
    const a = Math.abs(Math.log2((H[i] + 1e-8) / (P[i] + 1e-8)));
    const m = Math.min(1, Math.max(0, LIM + 0.5 - a));
    sl.real[i] *= m; sl.imag[i] *= m; sr.real[i] *= m; sr.imag[i] *= m;
  }
  return [istft(sl.real, sl.imag, F, B, FFT, HOP, xl.length), istft(sr.real, sr.imag, F, B, FFT, HOP, xr.length)];
}
function chunk(xl, xr) {
  const len = xl.length, L1 = FFT + Math.ceil((len + FFT) / HOP) * HOP;
  const pl = new Float32Array(L1), pr = new Float32Array(L1); pl.set(xl, FFT); pr.set(xr, FFT);
  const [a, b] = residual(pl, pr);
  return [a.subarray(FFT, FFT + len), b.subarray(FFT, FFT + len)];
}
export function extractAmbience(L, R, onProgress) {
  const N = L.length, C = SR * 20, PAD = SR * 2, XF = 2048, out = [new Float32Array(N), new Float32Array(N)];
  const K = Math.max(1, Math.ceil(N / C));
  for (let k = 0; k < K; k++) {
    const s = k * C, e = Math.min(N, s + C), a = Math.max(0, s - PAD), b = Math.min(N, e + PAD);
    const [rl, rr] = chunk(L.subarray(a, b), R.subarray(a, b));
    for (let t = Math.max(0, s - XF); t < Math.min(N, e + XF); t++) {
      let w = 1;
      if (k > 0 && t < s + XF) w = Math.min(w, (t - (s - XF)) / (2 * XF));
      if (k < K - 1 && t >= e - XF) w = Math.min(w, (e + XF - t) / (2 * XF));
      out[0][t] += rl[t - a] * w; out[1][t] += rr[t - a] * w;
    }
    onProgress?.((k + 1) / K);
  }
  return out;
}
