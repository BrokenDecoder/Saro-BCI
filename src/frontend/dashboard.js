/**
 * dashboard.js — Saro BCI Frontend Controller
 * =============================================
 * Improvements in this version:
 *  - Frequency band power strip (delta/theta/alpha/beta/gamma) live updates
 *  - Alert deduplication — same event text won't log twice in a row
 *  - Pause/resume of EEG rendering (Space key or button)
 *  - Keyboard shortcuts: Space=pause, R=reset 3D view, C=clear alerts
 *  - TDA warm-up indicator in System Status panel
 *  - Auto-rotate in 3D brain stops when OrbitControls detects user drag
 *  - Fixed `fetchChannels` undefined call removed
 *  - Band-dominant highlight: the most active band glows
 */
'use strict';

// ── Load msgpack-lite from CDN (in-browser msgpack decoder) ─────────────
(function injectMsgpackCDN() {
  const s = document.createElement('script');
  s.src   = 'https://cdn.jsdelivr.net/npm/msgpack-lite@0.1.26/dist/msgpack.min.js';
  s.onload = () => initDashboard();
  s.onerror = () => {
    console.warn('msgpack CDN failed — using JSON fallback mode');
    window._msgpackFallback = true;
    initDashboard();
  };
  document.head.appendChild(s);
})();

// ── Constants ────────────────────────────────────────────────────────────
const WS_URL          = `ws://${location.hostname}:8000/ws/eeg`;
const STATUS_URL      = `http://${location.hostname}:8000/status`;
const RECONNECT_DELAY = 3000;   // ms before reconnect attempt
const ALERT_THRESHOLD = 0.9;    // seizure probability threshold for banner
const MAX_LOG_ENTRIES = 100;    // cap alert log length

// Band names and their bar element IDs
const BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma'];
// Maximum expected RMS power per band (used for normalisation)
const BAND_MAX_RMS = { delta: 3.0, theta: 2.0, alpha: 1.5, beta: 1.0, gamma: 0.5 };

// Latency sparkline history
const LATENCY_HISTORY = new Array(60).fill(0);

// Audio: one AudioContext, created lazily on first user gesture
let _audioCtx = null;
let _audioMuted = false;

function _getAudioCtx() {
  if (!_audioCtx) {
    try { _audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch(_) {}
  }
  return _audioCtx;
}

/**
 * Play a clinical 3-beep alert (800→600→800 Hz, 120ms each).
 * Only fires for critical/high alerts and if not muted.
 */
function playAlertTone(severity) {
  if (_audioMuted) return;
  if (severity !== 'critical' && severity !== 'high') return;
  const ctx = _getAudioCtx();
  if (!ctx) return;
  const freqs = severity === 'critical' ? [880, 660, 880] : [660, 550, 660];
  let t = ctx.currentTime;
  freqs.forEach((freq, i) => {
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, t);
    gain.gain.setValueAtTime(0.0, t);
    gain.gain.linearRampToValueAtTime(0.18, t + 0.01);
    gain.gain.linearRampToValueAtTime(0.0,  t + 0.12);
    osc.start(t);
    osc.stop(t + 0.13);
    t += 0.16;
  });
}

function toggleMute() {
  _audioMuted = !_audioMuted;
  const btn = document.getElementById('btn-mute');
  if (btn) {
    btn.textContent = _audioMuted ? '🔇 Muted' : '🔔 Sound';
    btn.classList.toggle('muted', _audioMuted);
  }
}
window.toggleMute = toggleMute;

// ── State ────────────────────────────────────────────────────────────────
let ws           = null;
let eegRenderer  = null;
let tlRenderer   = null;
let brain3d      = null;
let frameCount   = 0;
let alertCount   = 0;
let connected    = false;
let statusTimer  = null;
let paused       = false;
let lastAlertMsg = '';          // for deduplication
let lastAlertTs  = 0;           // timestamp of last log entry

// ── Initialise everything once msgpack is ready ──────────────────────────
function initDashboard() {
  // Renderers
  eegRenderer = new EEGRenderer(document.getElementById('eeg-canvas'), {
    nChannels: 10,
    nSamples:  320,
    colors: ['#3b82f6','#06b6d4','#10b981','#8b5cf6','#f59e0b',
             '#ec4899','#f97316','#14b8a6','#a78bfa','#fb923c'],
  });
  tlRenderer = new TimelineRenderer(document.getElementById('timeline-canvas'));
  brain3d = new Brain3D('brain-container');

  // Slider controls
  document.getElementById('ch-slider').addEventListener('input', e => {
    eegRenderer.setChannels(parseInt(e.target.value));
  });
  document.getElementById('gain-slider').addEventListener('input', e => {
    eegRenderer.setGain(parseInt(e.target.value));
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', onKeyDown);

  // Kick off WebSocket + status polling
  connectWebSocket();
  fetchStatus();
  statusTimer = setInterval(fetchStatus, 5000);

  // Clock
  setInterval(updateClock, 1000);
  updateClock();
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────
function onKeyDown(e) {
  // Don't fire if typing in an input
  if (e.target.tagName === 'INPUT') return;

  switch (e.code) {
    case 'Space':
      e.preventDefault();
      togglePause();
      break;
    case 'KeyR':
      if (brain3d && brain3d.controls) {
        brain3d.controls.reset();
      }
      break;
    case 'KeyC':
      clearAlerts();
      break;
  }
}

// ── Pause / Resume ───────────────────────────────────────────────────────
function togglePause() {
  paused = !paused;
  const btn = document.getElementById('btn-pause');
  if (btn) {
    btn.textContent = paused ? '▶ Resume' : '⏸ Pause';
    btn.classList.toggle('paused', paused);
  }
}
window.togglePause = togglePause;

// ═══════════════════════════════════════════════════════════════════════════
// WebSocket — connect / reconnect
// ═══════════════════════════════════════════════════════════════════════════

function connectWebSocket() {
  setConnectionState(false);

  try {
    ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';
  } catch(e) {
    console.error('WebSocket create failed:', e);
    setTimeout(connectWebSocket, RECONNECT_DELAY);
    return;
  }

  ws.onopen = () => {
    console.log('[WS] Connected to', WS_URL);
    setConnectionState(true);
  };

  ws.onmessage = (event) => {
    let payload;
    try {
      if (window._msgpackFallback || typeof event.data === 'string') {
        payload = JSON.parse(event.data);
      } else {
        const buf = new Uint8Array(event.data);
        payload   = msgpack.decode(buf);
      }
    } catch(e) {
      console.warn('[WS] Decode error:', e);
      return;
    }
    handlePayload(payload);
  };

  ws.onerror = () => {
    console.warn('[WS] Error — will reconnect.');
    setConnectionState(false);
  };

  ws.onclose = () => {
    setConnectionState(false);
    console.log('[WS] Closed — reconnecting in', RECONNECT_DELAY, 'ms');
    setTimeout(connectWebSocket, RECONNECT_DELAY);
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Payload handler
// ═══════════════════════════════════════════════════════════════════════════

function handlePayload(p) {
  frameCount++;
  el('sys-frames').textContent = frameCount;

  if (p.type === 'alert') {
    handleAlert(p);
    return;
  }

  // ── Metrics ──────────────────────────────────────────────────────────

  const anomaly   = p.anomaly_score   ?? 0;
  const diverge   = p.divergence      ?? 0;
  const latency   = p.latency_ms      ?? 0;
  const mode      = p.mode            ?? 'stub';
  const fill      = p.buffer_fill     ?? 0;
  const isAnomaly = p.is_anomaly      ?? false;
  const nCh       = p.n_channels      ?? 128;
  const lslHw     = p.lsl_hardware    ?? false;
  const warmupPct = p.warmup_progress ?? 0;
  const isWarmed  = p.is_warmed_up    ?? false;

  // Total channels badge
  const chBadge = document.getElementById('total-channels');
  if (chBadge) chBadge.textContent = nCh;

  // LSL hardware indicator
  const lslBadge = document.getElementById('lsl-badge');
  if (lslBadge) lslBadge.style.display = lslHw ? 'flex' : 'none';

  // Anomaly score
  const aEl = el('anomaly-score');
  aEl.textContent = anomaly.toFixed(3);
  aEl.style.color = anomaly > 0.8 ? '#ef4444' : anomaly > 0.5 ? '#f59e0b' : '#10b981';
  barWidth('anomaly-bar', anomaly * 100);

  // Divergence (now a z-score; show raw value, normalise bar to [0,5])
  el('divergence-score').textContent = diverge.toFixed(2);
  barWidth('divergence-bar', Math.min(100, (diverge / 5) * 100));

  // Latency
  el('latency-ms').textContent = latency < 1 ? '<1 ms' : latency.toFixed(1) + ' ms';
  el('engine-label').textContent = 'Mode: ' + mode.toUpperCase();

  // Buffer fill
  el('buffer-fill').textContent = (fill * 100).toFixed(0) + '%';
  barWidth('buffer-bar', fill * 100);

  // Engine badge
  el('engine-mode').textContent = mode.toUpperCase();

  // ── TDA Warmup indicator ─────────────────────────────────────────────
  const warmEl = el('sys-warmup');
  if (warmEl) {
    if (isWarmed) {
      warmEl.textContent = 'Ready ✓';
      warmEl.classList.add('ready');
    } else {
      warmEl.textContent = `Baseline ${Math.round(warmupPct * 100)}%…`;
      warmEl.classList.remove('ready');
    }
  }

  // ── Band Power Strip ─────────────────────────────────────────────────
  if (p.band_power && Object.keys(p.band_power).length) {
    updateBandStrip(p.band_power);
  }

  // ── Timeline ──────────────────────────────────────────────────
  // Only push to timeline if warmed up; otherwise show 0
  tlRenderer.push(isWarmed ? anomaly : 0);

  // ── Latency sparkline ───────────────────────────────────────────
  LATENCY_HISTORY.push(latency);
  LATENCY_HISTORY.shift();
  drawLatencySparkline();

  // ── 3D Brain update ──────────────────────────────────────────────────
  if (brain3d && p.channel_power) {
    brain3d.updatePower(p.channel_power);
  }

  // ── EEG canvas ───────────────────────────────────────────────────────
  if (!paused && p.eeg_snippet && p.eeg_snippet.length > 0) {
    eegRenderer.setChannels(parseInt(el('ch-slider').value));
    eegRenderer.pushFrame(p.eeg_snippet);
  }

  // Body flash on critical anomaly (only when warmed up to avoid startup spam)
  if (isWarmed && isAnomaly && anomaly > ALERT_THRESHOLD) {
    document.body.classList.add('seizure-alert');
    setTimeout(() => document.body.classList.remove('seizure-alert'), 1600);
  }
}

// ── Band power strip ─────────────────────────────────────────────────────
function updateBandStrip(bandPower) {
  let maxRaw = 0, dominantBand = '';

  BANDS.forEach(band => {
    const raw  = bandPower[band] ?? 0;
    const cap  = BAND_MAX_RMS[band] ?? 1.0;
    const pct  = Math.min(100, (raw / cap) * 100);
    const barEl = el('bar-' + band);
    if (barEl) barEl.style.width = pct.toFixed(1) + '%';

    if (raw > maxRaw) { maxRaw = raw; dominantBand = band; }
  });

  // Highlight the dominant band
  BANDS.forEach(band => {
    const cell = el('band-' + band);
    if (cell) cell.classList.toggle('dominant', band === dominantBand);
  });
}

function handleAlert(a) {
  alertCount++;
  el('alert-count').textContent = alertCount;
  el('alert-count').parentElement.classList.add('has-alerts');
  el('last-alert-time').textContent = 'Last: ' + relTime(a.timestamp);

  // Banner
  const sev = (a.severity || 'medium').toLowerCase();
  showAlertBanner(sev, a.event, a.zone, a.score);

  // Audible alert tone
  playAlertTone(sev);

  // 3D Brain Alert
  if (brain3d) {
    brain3d.triggerAlert(a.zone);
  }

  // ── Deduplicated log entry ───────────────────────────────────
  const now = Date.now();
  const key = sev + '|' + (a.event || '');
  // Suppress if the same alert fired within the last 5 seconds
  if (key !== lastAlertMsg || (now - lastAlertTs) > 5000) {
    appendAlertLog(sev, a.event, a.score, a.timestamp);
    lastAlertMsg = key;
    lastAlertTs  = now;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Alert Banner
// ═══════════════════════════════════════════════════════════════════════════

let bannerTimer = null;

function showAlertBanner(severity, event, zone, score) {
  el('alert-banner').classList.add('visible');
  el('alert-severity').textContent = severity.toUpperCase();
  el('alert-message').textContent  = event + (score ? ` (score: ${score.toFixed(2)})` : '');
  el('alert-zone').textContent     = zone || 'Unknown Region';

  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(dismissAlert, 8000);
}

function dismissAlert() {
  el('alert-banner').classList.remove('visible');
}

window.dismissAlert = dismissAlert;

// ═══════════════════════════════════════════════════════════════════════════
// Alert Log
// ═══════════════════════════════════════════════════════════════════════════

function appendAlertLog(severity, event, score, timestamp) {
  const log   = el('alert-log');
  const empty = log.querySelector('.alert-log-empty');
  if (empty) empty.remove();

  const entry = document.createElement('div');
  entry.className = `alert-entry ${severity}`;
  entry.innerHTML = `
    <span class="entry-sev ${severity}">${severity.toUpperCase()}</span>
    <span class="entry-event">${event}</span>
    <span class="entry-score">${score ? score.toFixed(2) : '—'} &nbsp; ${relTime(timestamp)}</span>
  `;

  log.prepend(entry);

  const entries = log.querySelectorAll('.alert-entry');
  if (entries.length > MAX_LOG_ENTRIES) {
    entries[entries.length - 1].remove();
  }
}

function clearAlerts() {
  const log = el('alert-log');
  log.innerHTML = '<div class="alert-log-empty">No alerts yet — system is monitoring.</div>';
  alertCount = 0;
  el('alert-count').textContent = '0';
  el('alert-count').parentElement.classList.remove('has-alerts');
  el('last-alert-time').textContent = 'No alerts yet';
  lastAlertMsg = '';
  lastAlertTs  = 0;
}

window.clearAlerts = clearAlerts;

// ═══════════════════════════════════════════════════════════════════════════
// Status REST polling
// ═══════════════════════════════════════════════════════════════════════════

async function fetchStatus() {
  try {
    const res  = await fetch(STATUS_URL, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();

    el('sys-source').textContent   = data.data_source   ?? '—';
    el('sys-sfreq').textContent    = (data.sfreq_hz ?? '—') + ' Hz';
    el('sys-channels').textContent = (data.n_channels ?? '—') + ' ch';
    el('sys-uptime').textContent   = formatUptime(data.uptime_seconds ?? 0);
    el('sys-model').textContent    = 'EEGMamba SSM (' + (data.engine_mode ?? '?') + ')';
    el('client-count').textContent = data.clients ?? 0;

    if (!connected) setConnectionState(true);
  } catch (_) {
    // server not up yet — no-op
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Latency Sparkline
// ═══════════════════════════════════════════════════════════════════════════

function drawLatencySparkline() {
  let canvas = document.getElementById('latency-sparkline');
  if (!canvas) {
    // Create the sparkline canvas and inject it next to the latency metric
    const card = el('latency-ms');
    if (!card) return;
    canvas = document.createElement('canvas');
    canvas.id = 'latency-sparkline';
    canvas.style.cssText = 'width:100%;height:28px;display:block;margin-top:4px;';
    card.parentElement.appendChild(canvas);
  }
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = canvas.clientWidth  * dpr || 120 * dpr;
  canvas.height = 28 * dpr;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const maxVal = Math.max(...LATENCY_HISTORY, 10);
  const N = LATENCY_HISTORY.length;
  const stepX = W / (N - 1);

  ctx.beginPath();
  ctx.strokeStyle = '#60a5fa';
  ctx.lineWidth = 1.5 * dpr;
  LATENCY_HISTORY.forEach((v, i) => {
    const x = i * stepX;
    const y = H - (v / maxVal) * H * 0.9;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Max label
  ctx.fillStyle = 'rgba(100,116,139,0.8)';
  ctx.font = `${9 * dpr}px JetBrains Mono, monospace`;
  ctx.fillText(`max ${maxVal.toFixed(1)}ms`, 2 * dpr, 10 * dpr);
}

// ═══════════════════════════════════════════════════════════════════════════
// UI helpers
// ═══════════════════════════════════════════════════════════════════════════

function setConnectionState(isConnected) {
  connected = isConnected;
  const badge = el('connection-badge');
  const dot   = el('status-dot');
  const text  = el('status-text');

  badge.className = 'status-chip ' + (isConnected ? 'connected' : 'disconnected');
  dot.style.background = isConnected ? '#10b981' : '#ef4444';
  text.textContent = isConnected ? 'Live' : 'Reconnecting…';
}

function barWidth(id, pct) {
  el(id).style.width = Math.min(100, Math.max(0, pct)).toFixed(1) + '%';
}

function el(id) {
  return document.getElementById(id);
}

function updateClock() {
  const now = new Date();
  el('footer-time').textContent = now.toLocaleTimeString('en-GB', {
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  });
}

/**
 * Relative timestamp: "just now", "5s ago", "2m ago"
 */
function relTime(ts) {
  if (!ts) return '—';
  const secs = Math.round((Date.now() / 1000) - ts);
  if (secs < 5)  return 'just now';
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  return `${Math.floor(secs / 3600)}h ago`;
}

function formatTime(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleTimeString('en-GB', {
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  });
}

function formatUptime(secs) {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  return [h, m, s].map(v => String(v).padStart(2,'0')).join(':');
}
