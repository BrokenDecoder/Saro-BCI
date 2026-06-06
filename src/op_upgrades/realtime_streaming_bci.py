"""
OP Upgrade 1: Real-Time Closed-Loop Streaming
=============================================
Handles WebSocket connections to stream live EEG data from a headset and
simultaneously fetch the AI twin's latent state, computing a synchrony
score in real-time.

Usage (standalone):
    python realtime_streaming_bci.py              # default port 8765
    python realtime_streaming_bci.py --port 9000
"""

import asyncio
import logging
import argparse
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Placeholder stubs – replace with real implementations
# ---------------------------------------------------------------------------

def fetch_eeg_chunk() -> bytes:
    """Return the latest EEG chunk as raw bytes (stub).

    Replace with an LSL inlet pull or hardware SDK call.
    """
    import numpy as np
    chunk = np.random.randn(19, 25).astype("float32")   # 100 ms @ 250 Hz
    return chunk.tobytes()


def fetch_ai_state() -> bytes:
    """Return the current AI twin latent state as raw bytes (stub).

    Replace with the actual EEGMambaTwin.predict_healthy_state() output.
    """
    import numpy as np
    state = np.random.randn(3).astype("float32")        # 3-D CEBRA embedding
    return state.tobytes()


def calculate_cebra_synchrony(eeg_chunk: bytes, ai_state: bytes) -> float:
    """Compute cosine similarity between EEG chunk and AI state (stub).

    Replace with real CEBRA-based alignment score.
    """
    import numpy as np
    eeg = np.frombuffer(eeg_chunk, dtype="float32")
    state = np.frombuffer(ai_state, dtype="float32")
    # Pad / truncate to the same length for the stub
    n = min(len(eeg), len(state))
    a, b = eeg[:n], state[:n]
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / norm)


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def stream_bci_data(websocket, path: str = "/") -> None:
    """Handle a single WebSocket client connection.

    Continuously fetches an EEG chunk, computes the synchrony score with
    the AI twin, and pushes the result back to the client.

    Parameters
    ----------
    websocket : websockets.WebSocketServerProtocol
    path      : URL path (required by legacy websockets API; ignored here).
    """
    peer = websocket.remote_address
    logger.info("Client connected from %s (path=%s)", peer, path)

    try:
        while True:
            eeg_chunk = fetch_eeg_chunk()
            ai_state  = fetch_ai_state()
            sync_score = calculate_cebra_synchrony(eeg_chunk, ai_state)

            message = f"SYNC_SCORE:{sync_score:.6f}"
            await websocket.send(message)
            logger.debug("Sent %s to %s", message, peer)

            # ~100 ms cadence – adjust to match your EEG chunk size
            await asyncio.sleep(0.1)

    except Exception as exc:  # catches websockets.ConnectionClosed among others
        logger.info("Connection with %s closed: %s", peer, exc)


# ---------------------------------------------------------------------------
# Server entry-point
# ---------------------------------------------------------------------------

async def _run_server(host: str = "localhost", port: int = 8765) -> None:
    try:
        import websockets  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "websockets is required for real-time streaming. "
            "Install it with: pip install websockets"
        ) from e

    logger.info("Starting BCI WebSocket server on ws://%s:%d", host, port)
    async with websockets.serve(stream_bci_data, host, port):
        await asyncio.Future()   # run forever


def start_server(host: str = "localhost", port: int = 8765) -> None:
    """Blocking entry-point – runs the WebSocket server until interrupted."""
    asyncio.run(_run_server(host=host, port=port))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Saro BCI real-time WebSocket server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    start_server(host=args.host, port=args.port)
