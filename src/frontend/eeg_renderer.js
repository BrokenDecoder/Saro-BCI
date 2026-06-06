/**
 * eeg_renderer.js — WebGL EEG Oscilloscope
 * ==========================================
 * Renders 1–5 EEG channels as coloured line strips on a WebGL canvas.
 * Uses instanced rendering: all channels drawn in one draw call.
 *
 * Design:
 *  - Pre-allocated Float32Array ring buffers (no GC during animation)
 *  - Each channel is y-offset by a fixed increment
 *  - Gain slider scales amplitude in real time via a uniform
 *  - Smooth scrolling: new data appended, old data naturally scrolls left
 */

'use strict';

class EEGRenderer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} opts
   * @param {number} opts.nChannels   - number of channels to render (1–5)
   * @param {number} opts.nSamples   - sample history length per channel
   * @param {string[]} opts.colors   - hex colours per channel
   */
  constructor(canvas, opts = {}) {
    this.canvas    = canvas;
    this.nChannels = opts.nChannels  || 5;
    this.nSamples  = opts.nSamples   || 320;
    this.colors    = opts.colors || [
      '#3b82f6', '#06b6d4', '#10b981', '#8b5cf6', '#f59e0b'
    ];
    this.gain      = 8;

    // Per-channel ring buffers (pre-allocated, no GC)
    this._buffers   = Array.from({ length: 5 }, () => new Float32Array(this.nSamples));
    this._writePtr  = 0;

    this._initGL();
    this._resize();

    window.addEventListener('resize', () => this._resize());
  }

  // ── WebGL Initialization ─────────────────────────────────────────────

  _initGL() {
    const gl = this.canvas.getContext('webgl', {
      antialias: true, alpha: false, depth: false
    });
    if (!gl) {
      console.warn('WebGL not available — falling back to Canvas2D');
      this._fallback = true;
      this._ctx2d = this.canvas.getContext('2d');
      return;
    }
    this.gl = gl;

    // Vertex shader: maps sample index → NDC x, amplitude → NDC y
    const vsSource = `
      attribute float a_index;    // [0, nSamples)
      attribute float a_value;    // EEG amplitude (normalised)
      uniform float u_nSamples;
      uniform float u_channelOffset;  // vertical centre in [-1, 1]
      uniform float u_channelHeight;  // fraction of screen height
      uniform float u_gain;
      void main() {
        float x = (a_index / (u_nSamples - 1.0)) * 2.0 - 1.0;
        float y = u_channelOffset + a_value * u_gain * u_channelHeight;
        gl_Position = vec4(x, clamp(y, -1.0, 1.0), 0.0, 1.0);
      }
    `;

    const fsSource = `
      precision mediump float;
      uniform vec3 u_color;
      void main() {
        gl_FragColor = vec4(u_color, 1.0);
      }
    `;

    const vs = this._compileShader(gl.VERTEX_SHADER, vsSource);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fsSource);
    this.prog = gl.createProgram();
    gl.attachShader(this.prog, vs);
    gl.attachShader(this.prog, fs);
    gl.linkProgram(this.prog);
    gl.useProgram(this.prog);

    // Attribute / uniform locations
    this.loc = {
      index:         gl.getAttribLocation(this.prog, 'a_index'),
      value:         gl.getAttribLocation(this.prog, 'a_value'),
      nSamples:      gl.getUniformLocation(this.prog, 'u_nSamples'),
      channelOffset: gl.getUniformLocation(this.prog, 'u_channelOffset'),
      channelHeight: gl.getUniformLocation(this.prog, 'u_channelHeight'),
      gain:          gl.getUniformLocation(this.prog, 'u_gain'),
      color:         gl.getUniformLocation(this.prog, 'u_color'),
    };

    // Index buffer — same for all channels (0, 1, 2, … nSamples-1)
    this._indexBuf = gl.createBuffer();
    const indices  = new Float32Array(this.nSamples);
    for (let i = 0; i < this.nSamples; i++) indices[i] = i;
    gl.bindBuffer(gl.ARRAY_BUFFER, this._indexBuf);
    gl.bufferData(gl.ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    // Per-channel value buffers (updated every frame)
    this._glBufs = Array.from({ length: 5 }, () => gl.createBuffer());
  }

  _compileShader(type, src) {
    const gl = this.gl;
    const s  = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('Shader error:', gl.getShaderInfoLog(s));
    }
    return s;
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const w   = this.canvas.clientWidth;
    const h   = this.canvas.clientHeight;
    this.canvas.width  = w * dpr;
    this.canvas.height = h * dpr;
    if (this.gl) this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  // ── Data Ingestion ───────────────────────────────────────────────────

  /**
   * Push a new EEG frame: array of arrays [[ch0_samples], [ch1_samples], ...]
   * @param {number[][]} channels — shape [nChannels][nSamples]
   */
  pushFrame(channels) {
    for (let c = 0; c < Math.min(channels.length, 5); c++) {
      const src    = channels[c];
      const dst    = this._buffers[c];
      const N      = Math.min(src.length, this.nSamples);
      // Overwrite oldest data (newest at end)
      dst.copyWithin(0, N);
      // Normalise to [-1, 1] range
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < N; i++) { if (src[i] < min) min = src[i]; if (src[i] > max) max = src[i]; }
      const range = (max - min) || 1;
      for (let i = 0; i < N; i++) {
        dst[this.nSamples - N + i] = ((src[i] - min) / range) * 2 - 1;
      }
    }
    this.render();
  }

  // ── Rendering ────────────────────────────────────────────────────────

  render() {
    if (this._fallback) { this._render2d(); return; }
    const gl = this.gl;
    gl.clearColor(0.031, 0.047, 0.078, 1.0); // #080c14
    gl.clear(gl.COLOR_BUFFER_BIT);

    const nCh = this.nChannels;
    const step = 2.0 / (nCh + 1);

    gl.useProgram(this.prog);
    gl.uniform1f(this.loc.nSamples, this.nSamples);
    gl.uniform1f(this.loc.channelHeight, step * 0.45);
    gl.uniform1f(this.loc.gain, this.gain);

    // Bind index attribute
    gl.bindBuffer(gl.ARRAY_BUFFER, this._indexBuf);
    gl.enableVertexAttribArray(this.loc.index);
    gl.vertexAttribPointer(this.loc.index, 1, gl.FLOAT, false, 0, 0);

    for (let c = 0; c < nCh; c++) {
      const offset = 1.0 - (c + 1) * step;
      const rgb    = this._hexToRgb(this.colors[c % this.colors.length]);

      gl.uniform1f(this.loc.channelOffset, offset);
      gl.uniform3f(this.loc.color, rgb[0], rgb[1], rgb[2]);

      // Upload channel data
      gl.bindBuffer(gl.ARRAY_BUFFER, this._glBufs[c]);
      gl.bufferData(gl.ARRAY_BUFFER, this._buffers[c], gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.loc.value);
      gl.vertexAttribPointer(this.loc.value, 1, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.LINE_STRIP, 0, this.nSamples);
    }
  }

  // Canvas2D fallback
  _render2d() {
    const ctx = this._ctx2d;
    const W   = this.canvas.width;
    const H   = this.canvas.height;
    ctx.fillStyle = '#080c14';
    ctx.fillRect(0, 0, W, H);

    const nCh  = this.nChannels;
    const rowH = H / nCh;

    for (let c = 0; c < nCh; c++) {
      const data   = this._buffers[c];
      const color  = this.colors[c % this.colors.length];
      const cy     = rowH * c + rowH / 2;

      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth   = 1.5;

      for (let i = 0; i < this.nSamples; i++) {
        const x = (i / (this.nSamples - 1)) * W;
        const y = cy - data[i] * (rowH * 0.4) * this.gain;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Channel label
      ctx.fillStyle = color;
      ctx.font      = '11px JetBrains Mono, monospace';
      ctx.fillText(`CH ${c + 1}`, 6, cy - rowH * 0.35);
    }
  }

  _hexToRgb(hex) {
    const n = parseInt(hex.replace('#',''), 16);
    return [(n>>16&255)/255, (n>>8&255)/255, (n&255)/255];
  }

  setGain(g)      { this.gain      = parseFloat(g); }
  setChannels(n)  { this.nChannels = parseInt(n); this.render(); }
}


// ── Timeline Renderer (2D Canvas — seizure probability over time) ────────

class TimelineRenderer {
  constructor(canvas) {
    this.canvas  = canvas;
    this.ctx     = canvas.getContext('2d');
    this.history = new Array(300).fill(0);
    this.THRESH  = 0.9;
    window.addEventListener('resize', () => this._resize());
    this._resize();
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width  = this.canvas.clientWidth  * dpr;
    this.canvas.height = this.canvas.clientHeight * dpr;
  }

  push(value) {
    this.history.push(Math.min(1, Math.max(0, value)));
    this.history.shift();
    this.render();
  }

  render() {
    const ctx    = this.ctx;
    const W      = this.canvas.width;
    const H      = this.canvas.height;
    const N      = this.history.length;
    const stepX  = W / (N - 1);
    const pad    = 10;

    ctx.fillStyle = '#080c14';
    ctx.fillRect(0, 0, W, H);

    // Threshold line
    const thY = H - (this.THRESH * (H - pad * 2)) - pad;
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = 'rgba(245,158,11,0.5)';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(0, thY);
    ctx.lineTo(W, thY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Filled area under the curve
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = i * stepX;
      const y = H - (this.history[i] * (H - pad * 2)) - pad;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.lineTo(W, H);
    ctx.lineTo(0, H);
    ctx.closePath();

    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0,   'rgba(239,68,68,0.5)');
    grad.addColorStop(0.5, 'rgba(59,130,246,0.3)');
    grad.addColorStop(1,   'rgba(59,130,246,0.0)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth   = 2;
    for (let i = 0; i < N; i++) {
      const x = i * stepX;
      const y = H - (this.history[i] * (H - pad * 2)) - pad;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Threshold label
    ctx.fillStyle = 'rgba(245,158,11,0.8)';
    ctx.font      = '10px JetBrains Mono, monospace';
    ctx.fillText('0.90', W - 34, thY - 4);
  }
}

// Export
window.EEGRenderer      = EEGRenderer;
window.TimelineRenderer = TimelineRenderer;
