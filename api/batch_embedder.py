"""
batch_embedder.py
-----------------
SharedBatchEmbedder — a singleton that collects face-crop tensors submitted by
any number of CameraManager instances and runs them through the embedding model
as a single batched GPU forward pass.

Why this helps
--------------
Each camera's ML thread submits crops one at a time.  With N cameras each
producing 1-2 faces per frame, the naive approach calls model(tensor) N×2
times per second — small batches (size 1) that leave 90 %+ of GPU idle.

SharedBatchEmbedder accumulates submissions for at most `wait_ms` milliseconds
(default 8 ms — comfortably less than one camera frame interval at 30 fps),
then stacks them into a single (B, C, H, W) tensor and runs one forward pass.
Total latency added per crop is ≤ wait_ms.  On CPU the overhead is minimal
but the code path is identical, so no special-casing is needed.

Usage (inside CameraManager)
-----------------------------
    from api.batch_embedder import get_batch_embedder
    emb = get_batch_embedder().embed_sync(tensor_1chw)   # blocks ≤ wait_ms
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import torch


class SharedBatchEmbedder:
    """
    Parameters
    ----------
    model   : the embedding model (nn.Module); must already be on the right device.
    device  : torch.device
    wait_ms : how long (milliseconds) to collect crops before flushing the batch.
              Lower → less latency, smaller batches.
              Higher → bigger batches, better GPU utilisation.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 wait_ms: float = 8.0):
        self.model   = model
        self.device  = device
        self.wait_ms = wait_ms

        # Each entry: (tensor_1chw, result_list, ready_event)
        self._queue:  list = []
        self._q_lock  = threading.Lock()

        # Metrics: batch sizes seen (EWA)
        self._avg_batch: float = 1.0

        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="batch-embedder")
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_sync(self, tensor: torch.Tensor,
                   timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Submit one (1, C, H, W) or (C, H, W) tensor and block until the
        embedding is ready.  Returns a (D,) float32 numpy array, or None on
        timeout / error.
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)   # → (1, C, H, W)

        result: list  = []           # worker writes [emb_ndarray] here
        ready         = threading.Event()

        with self._q_lock:
            self._queue.append((tensor, result, ready))

        if ready.wait(timeout):
            return result[0] if result else None
        return None

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker(self):
        while not self._stop.is_set():
            # Wait for at least one item
            time.sleep(self.wait_ms / 1000.0)

            with self._q_lock:
                if not self._queue:
                    continue
                batch_items = self._queue[:]
                self._queue.clear()

            # Stack into one batch tensor
            tensors = torch.cat([item[0] for item in batch_items], dim=0).to(self.device)

            # EWA batch size (for metrics)
            bs = len(batch_items)
            self._avg_batch = 0.9 * self._avg_batch + 0.1 * bs

            try:
                with torch.no_grad():
                    embs = self.model(tensors)   # (B, D)
                if torch.is_tensor(embs):
                    embs = embs.detach().cpu()
            except Exception as exc:
                # On error push None to all waiters
                for (_, result, ready) in batch_items:
                    result.clear()
                    ready.set()
                print(f"[BatchEmbedder] inference error: {exc}")
                continue

            # Distribute results
            for i, (_, result, ready) in enumerate(batch_items):
                emb = embs[i].numpy().astype(np.float32)
                result.append(emb)
                ready.set()

    # ── Info ──────────────────────────────────────────────────────────────────

    @property
    def avg_batch_size(self) -> float:
        return round(self._avg_batch, 2)


# ── Singleton management ──────────────────────────────────────────────────────

_instance: Optional[SharedBatchEmbedder] = None
_init_lock = threading.Lock()


def get_batch_embedder(model=None, device=None,
                       wait_ms: float = 8.0) -> SharedBatchEmbedder:
    """
    Return the process-wide SharedBatchEmbedder, creating it on first call.
    Pass `model` and `device` only on the first call (or to replace the
    existing instance with a new one).
    """
    global _instance
    with _init_lock:
        if _instance is None:
            if model is None or device is None:
                raise RuntimeError(
                    "get_batch_embedder: model and device are required on first call.")
            _instance = SharedBatchEmbedder(model=model, device=device,
                                            wait_ms=wait_ms)
        elif model is not None:
            # Caller wants to (re-)initialise with a new model
            _instance.stop()
            _instance = SharedBatchEmbedder(model=model, device=device,
                                            wait_ms=wait_ms)
    return _instance


def stop_batch_embedder():
    global _instance
    with _init_lock:
        if _instance is not None:
            _instance.stop()
            _instance = None