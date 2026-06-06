"""
Phase 3 – WebSocket Streaming Backend
======================================
FastAPI server that:
  1. Starts the Phase 2 BCIPipeline in background threads
  2. Feeds it from PhysioNet replay (real EEG) or LSL inlet
  3. Broadcasts results to ALL connected WebSocket clients at ~30 FPS
  4. Serves the frontend dashboard as static files at /
  5. Serves REST endpoints for status, config, and alert history

Run with:
    python src/backend/server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Set

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# ── Project path setup ────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.pipeline import BCIPipeline, PipelineConfig
from src.core.lsl_bridge import LSLBridge
from src.preprocessing.physionet_fetcher import fetch_real_eeg_data

logger = logging.getLogger("saro.backend")

# ── Constants ─────────────────────────────────────────────────────────────
BROADCAST_FPS     = 30
BROADCAST_PERIOD  = 1.0 / BROADCAST_FPS
ALERT_HISTORY_MAX = 200
PHYSIONET_SUBJECT = 1
PHYSIONET_RUN     = 3
N_CHANNELS        = 128   # ADS1299 x8 target
CHUNK_SIZE        = 25    # samples per push (100ms at 250 Hz)


# ═══════════════════════════════════════════════════════════════════════════
# Connection Manager — tracks all live WebSocket clients
# ═══════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        logger.info("Client connected. Total: %d", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(ws)
        logger.info("Client disconnected. Total: %d", len(self._connections))

    async def broadcast_bytes(self, data: bytes) -> None:
        """Send bytes to all connected clients, drop dead ones silently."""
        if not self._connections:
            return
        dead = set()
        async with self._lock:
            targets = list(self._connections)
        for ws in targets:
            try:
                await ws.send_bytes(data)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self._connections -= dead

    @property
    def count(self) -> int:
        return len(self._connections)


# ═══════════════════════════════════════════════════════════════════════════
# Application State
# ═══════════════════════════════════════════════════════════════════════════

class AppState:
    """Shared mutable state for the whole server lifecycle."""
    pipeline:       Optional[BCIPipeline]    = None
    lsl_bridge:     Optional[LSLBridge]      = None
    manager:        Optional[ConnectionManager] = None
    result_queue:   asyncio.Queue            = None   # type: ignore
    alert_history:  deque                   = deque(maxlen=ALERT_HISTORY_MAX)
    latest_result:  dict                    = {}
    data_source:    str                     = "physionet"  # or "lsl" or "synthetic"
    eeg_data:       Optional[np.ndarray]    = None
    eeg_sfreq:      float                   = 160.0
    replay_idx:     int                     = 0
    is_ready:       bool                    = False
    start_time:     float                   = time.time()
    lsl_hardware:   bool                    = False   # True if real hw connected
    n_channels:     int                     = N_CHANNELS


state = AppState()


# ═══════════════════════════════════════════════════════════════════════════
# Data source: PhysioNet replay
# ═══════════════════════════════════════════════════════════════════════════

def _physionet_replay_thread() -> None:
    """Thread: load 128-ch PhysioNet data and replay into the pipeline."""
    logger.info("Loading PhysioNet EEG data (padded to %dch)...", N_CHANNELS)
    try:
        data, sfreq = fetch_real_eeg_data(
            subject=PHYSIONET_SUBJECT,
            run=PHYSIONET_RUN,
            tmin=0.0,
            tmax=120.0,
            target_channels=N_CHANNELS,
        )
        state.eeg_data  = data
        state.eeg_sfreq = sfreq
        logger.info(
            "PhysioNet data loaded: shape=%s sfreq=%dHz",
            data.shape, sfreq
        )
    except Exception as exc:
        logger.warning("PhysioNet load failed (%s) — using synthetic data.", exc)
        sfreq = 250
        state.eeg_data  = np.random.randn(N_CHANNELS, int(120 * sfreq)).astype(np.float32)
        state.eeg_sfreq = float(sfreq)

    state.is_ready = True
    sleep_per_chunk = CHUNK_SIZE / state.eeg_sfreq
    total_samples   = state.eeg_data.shape[1]

    logger.info("PhysioNet replay started (%.1f Hz, %d samples, %d channels)",
                state.eeg_sfreq, total_samples, N_CHANNELS)

    while True:  # loop the recording forever
        start = state.replay_idx
        end   = start + CHUNK_SIZE
        if end > total_samples:
            state.replay_idx = 0
            continue

        chunk = state.eeg_data[:, start:end].copy()
        if state.pipeline:
            state.pipeline.push_chunk(chunk)

        state.replay_idx = end
        time.sleep(sleep_per_chunk)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline callbacks (run in inference thread, bridge to async queue)
# ═══════════════════════════════════════════════════════════════════════════

def _on_result(result: dict) -> None:
    """Called by BCIPipeline on every inference step (background thread)."""
    payload = {
        "type":             "result",
        "timestamp":        result.get("timestamp", time.time()),
        "anomaly_score":    float(result.get("anomaly_score", 0.0)),
        "divergence":       float(result.get("divergence_score", 0.0)),
        "is_anomaly":       bool(result.get("is_anomaly", False)),
        "latency_ms":       float(result.get("latency_ms", 0.0)),
        "mode":             result.get("mode", "stub"),
        "buffer_fill":      float(result.get("buffer_fill", 0.0)),
        "n_channels":       N_CHANNELS,
        "lsl_hardware":     state.lsl_hardware,
        "warmup_progress":  float(result.get("warmup_progress", 0.0)),
        "is_warmed_up":     bool(result.get("is_warmed_up", False)),
        # Band power per EEG band (dict: delta/theta/alpha/beta/gamma)
        "band_power":       result.get("band_power", {}),
        # EEG snippet (10 evenly-spaced channels from 128)
        "eeg_snippet":      _get_eeg_snippet(),
        # RMS power per channel for 3D brain map
        "channel_power":    _get_channel_power(),
    }
    state.latest_result = payload

    # Bridge from sync thread → async event loop
    if state.result_queue:
        try:
            state.result_queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # drop silently — broadcast loop will use latest_result


def _on_alert(alert_json: str, result: dict) -> None:
    """Called by BCIPipeline on anomaly detection (background thread)."""
    try:
        alert_data = json.loads(alert_json)
    except Exception:
        alert_data = {"raw": alert_json}

    alert_payload = {
        "type":       "alert",
        "timestamp":  time.time(),
        "severity":   alert_data.get("severity_level", "unknown"),
        "score":      float(result.get("divergence_score", 0.0)),
        "event":      alert_data.get("event", "Anomaly Detected"),
        "zone":       alert_data.get("seizure_onset_zone", "Unknown"),
    }
    state.alert_history.appendleft(alert_payload)
    logger.warning(
        "[ALERT] %s severity=%-8s score=%.2f",
        alert_payload["event"],
        alert_payload["severity"].upper(),
        alert_payload["score"],
    )

    if state.result_queue:
        try:
            state.result_queue.put_nowait(alert_payload)
        except asyncio.QueueFull:
            pass


def _get_eeg_snippet(n_plot: int = 10, n_display_samples: int = 128) -> list:
    """Return the last n_display_samples of n_plot evenly-spaced channels.

    Downsampled from 320 → 128 samples to reduce WebSocket payload by 2.5×
    with no visible quality loss at the display pixel density (~320px wide).
    """
    if state.pipeline is None:
        return []
    try:
        window = state.pipeline.ring_buffer.read_window(window_samples=320)
        n_ch = window.shape[0]
        # Pick n_plot evenly-spaced channel indices across the full array
        idxs = np.linspace(0, n_ch - 1, min(n_plot, n_ch), dtype=int)
        snippet = window[idxs, :]   # (n_plot, 320)
        # Downsample time axis to n_display_samples
        if snippet.shape[1] > n_display_samples:
            step = snippet.shape[1] // n_display_samples
            snippet = snippet[:, ::step][:, :n_display_samples]
        return snippet.tolist()
    except Exception:
        return []

def _get_channel_power() -> list[float]:
    """Return the RMS power for all channels over the last 320 samples."""
    if state.pipeline is None:
        return []
    try:
        window = state.pipeline.ring_buffer.read_window(window_samples=320)
        # Compute RMS: sqrt(mean(window^2)) across the time axis (axis=1)
        rms = np.sqrt(np.mean(window ** 2, axis=1))
        # Add a tiny epsilon to avoid absolute zeros, convert to float list
        return (rms + 1e-6).tolist()
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI application lifecycle
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: launch pipeline + replay. Shutdown: stop gracefully."""
    logger.info("Saro BCI backend starting up...")

    # 1. Async queue (bridge between threads and async broadcast loop)
    state.result_queue = asyncio.Queue(maxsize=256)

    # 2. Connection manager
    state.manager = ConnectionManager()

    # 3. Build and start the BCIPipeline at 128ch
    cfg = PipelineConfig()
    cfg.n_channels   = N_CHANNELS
    cfg.weights_path = str(_PROJECT_ROOT / "pretrained_weights" / "pretrained_EEGMamba.pth")
    cfg.onnx_path    = str(_PROJECT_ROOT / "pretrained_weights" / "eegmamba.onnx")
    cfg.buffer_seconds = 5.0

    state.n_channels = N_CHANNELS
    state.pipeline = BCIPipeline(
        config=cfg,
        on_result=_on_result,
        on_alert=_on_alert,
    )
    state.pipeline.start()

    # 4. Try LSL hardware first; fall back to PhysioNet replay
    state.lsl_bridge = LSLBridge(n_channels=N_CHANNELS, sfreq=cfg.sfreq,
                                  chunk_size=CHUNK_SIZE)
    lsl_ok = state.lsl_bridge.connect(timeout=3.0)

    if lsl_ok:
        state.lsl_hardware = True
        state.data_source  = "lsl_hardware"
        state.is_ready     = True
        state.lsl_bridge.start(state.pipeline)
        logger.info("LSL hardware connected — streaming 128ch real EEG.")
    else:
        state.lsl_hardware = False
        state.data_source  = "physionet_128ch"
        # PhysioNet replay thread
        replay_thread = threading.Thread(
            target=_physionet_replay_thread, daemon=True, name="physionet-replay"
        )
        replay_thread.start()

    # 5. Broadcast loop (async task)
    broadcast_task = asyncio.create_task(_broadcast_loop())

    logger.info("Saro BCI 128ch backend ready on http://0.0.0.0:8000")
    yield  # <- server runs here

    # Shutdown
    broadcast_task.cancel()
    if state.lsl_bridge:
        state.lsl_bridge.stop()
    if state.pipeline:
        state.pipeline.stop()
    logger.info("Saro BCI backend shut down.")


async def _broadcast_loop() -> None:
    """Async task: drain result_queue and broadcast to all WebSocket clients."""
    while True:
        try:
            payload = await asyncio.wait_for(
                state.result_queue.get(), timeout=BROADCAST_PERIOD
            )
        except asyncio.TimeoutError:
            # No new data — broadcast the latest result anyway to keep FPS
            payload = state.latest_result
            if not payload:
                continue

        if state.manager and state.manager.count > 0:
            packed = msgpack.packb(payload, use_bin_type=True)
            await state.manager.broadcast_bytes(packed)


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════════════════════

_FRONTEND_DIR = _PROJECT_ROOT / "src" / "frontend"

app = FastAPI(
    title="Saro BCI Neural Twin Backend",
    description="Real-time EEG processing and AI Twin WebSocket streaming API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend dashboard at /dashboard/
if _FRONTEND_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")

@app.get("/")
async def root():
    """Redirect root to the frontend dashboard."""
    return RedirectResponse(url="/dashboard/")


# ── WebSocket endpoint ─────────────────────────────────────────────────────

@app.websocket("/ws/eeg")
async def websocket_eeg(ws: WebSocket) -> None:
    """
    Main WebSocket endpoint. Clients connect here to receive real-time
    EEG data and AI Twin predictions as MessagePack-encoded binary frames.

    Frame schema (msgpack-decoded):
    {
      "type":          "result" | "alert",
      "timestamp":     float,          # Unix timestamp
      "anomaly_score": float,          # [0, 1]
      "divergence":    float,          # raw Wasserstein divergence
      "is_anomaly":    bool,
      "latency_ms":    float,          # inference latency
      "mode":          str,            # "onnx" | "pytorch" | "stub"
      "buffer_fill":   float,          # ring buffer fill ratio [0, 1]
      "eeg_snippet":   [[float, ...]], # 5 channels × 320 samples
      # alert-only:
      "severity":      str,            # "low" | "medium" | "high" | "critical"
      "event":         str,
      "zone":          str,
    }
    """
    await state.manager.connect(ws)
    try:
        while True:
            # Keep connection alive — we push data, client can send config
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                # Handle client → server config messages (future use)
                logger.debug("Client message: %s", msg)
            except asyncio.TimeoutError:
                pass  # no client message — normal, we just keep pushing
    except WebSocketDisconnect:
        pass
    finally:
        await state.manager.disconnect(ws)


# ── REST endpoints ─────────────────────────────────────────────────────────

@app.get("/status")
async def get_status() -> JSONResponse:
    """System health and pipeline status."""
    det = state.pipeline.detector if state.pipeline else None
    return JSONResponse({
        "status":              "running",
        "uptime_seconds":      round(time.time() - state.start_time, 1),
        "data_source":         state.data_source,
        "lsl_hardware":        state.lsl_hardware,
        "is_ready":            state.is_ready,
        "clients":             state.manager.count if state.manager else 0,
        "engine_mode":         state.pipeline.engine._mode if state.pipeline else "offline",
        "buffer_fill":         round(
            state.pipeline.ring_buffer.fill_ratio if state.pipeline else 0.0, 3
        ),
        "n_channels":          state.n_channels,
        "sfreq_hz":            state.pipeline.config.sfreq if state.pipeline else 0,
        "alert_count":         len(state.alert_history),
        "warmup_progress":     round(det.warmup_progress, 3) if det else 0.0,
        "is_warmed_up":        det.is_warmed_up if det else False,
        "z_score_threshold":   2.5,
    })


@app.get("/alerts")
async def get_alerts(limit: int = 50) -> JSONResponse:
    """Return the most recent `limit` alerts."""
    alerts = list(state.alert_history)[:limit]
    return JSONResponse({"alerts": alerts, "total": len(state.alert_history)})


@app.get("/latest")
async def get_latest() -> JSONResponse:
    """Return the latest inference result snapshot (for REST polling fallback)."""
    result = dict(state.latest_result)
    # Omit the large eeg_snippet from REST (WebSocket only)
    result.pop("eeg_snippet", None)
    return JSONResponse(result or {"status": "no data yet"})


@app.get("/config")
async def get_config() -> JSONResponse:
    """Return current pipeline configuration."""
    if not state.pipeline:
        return JSONResponse({"error": "pipeline not initialised"}, status_code=503)
    cfg = state.pipeline.config
    det = state.pipeline.detector
    return JSONResponse({
        "n_channels":          cfg.n_channels,
        "sfreq_hz":            cfg.sfreq,
        "buffer_seconds":      cfg.buffer_seconds,
        "window_seconds":      cfg.window_seconds,
        "step_seconds":        cfg.step_seconds,
        "lowcut_hz":           cfg.lowcut,
        "highcut_hz":          cfg.highcut,
        "notch_hz":            cfg.notch_freq,
        # TDA threshold (replaces old divergence_threshold which was hardcoded 0.8)
        "z_score_threshold":   2.5,
        "tda_warmup_n":        20,
        "tda_warmup_progress": round(det.warmup_progress, 3),
        "tda_warmed_up":       det.is_warmed_up,
    })


@app.get("/channels")
async def get_channels() -> JSONResponse:
    """Return the 128-channel label list (ADS1299 10-20 layout)."""
    if state.lsl_bridge:
        labels = state.lsl_bridge.get_channel_labels()
    else:
        labels = [f"CH{i+1:03d}" for i in range(N_CHANNELS)]
    return JSONResponse({"n_channels": N_CHANNELS, "labels": labels})


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


# ── Dev server entry-point ─────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    )
    uvicorn.run(
        "src.backend.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
