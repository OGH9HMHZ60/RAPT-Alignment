"""Feature extraction for symbolic music alignment.

Provides utilities for converting symbolic note arrays into pianoroll
representations, extracting fixed-size temporal windows, applying tempo
resampling, and running batched inference through a Transformer ensemble.
"""

import threading
from typing import List, Tuple

import numpy as np
import partitura as pt
import torch


# Default temporal resolution (frames per second) for pianoroll quantization.
TIME_DIV = 50

# Module-level cache keyed on (id, length, time_div) of the input note array.
_pianoroll_cache: dict = {}


def clear_pianoroll_cache() -> None:
    """Empty the pianoroll cache."""
    global _pianoroll_cache
    _pianoroll_cache = {}


def compute_pianoroll_88(
    note_array: np.ndarray,
    time_div: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an 89-dimensional pianoroll from a symbolic note array.

    The first 88 channels encode the standard piano range as a binary pianoroll
    in which onset frames are scaled to 5.0 to emphasise note attacks. The 89th
    channel is a global onset indicator.

    Args:
        note_array: Structured note array (Partitura convention).
        time_div: Frames per second. Defaults to ``TIME_DIV``.

    Returns:
        A tuple ``(pianoroll, idx)`` where ``pianoroll`` has shape
        ``(num_frames, 89)`` and ``idx`` maps each note to its active frame
        bounds.
    """
    if time_div is None:
        time_div = TIME_DIV

    cache_key = (id(note_array), len(note_array), time_div)
    if cache_key in _pianoroll_cache:
        return _pianoroll_cache[cache_key]

    pr_sparse, idx = pt.utils.music.compute_pianoroll(
        note_info=note_array,
        return_idxs=True,
        piano_range=True,
        time_div=time_div,
        binary=True,
    )
    pr = pr_sparse.toarray().T.astype(np.float32)

    onset_channel = np.zeros((pr.shape[0], 1), dtype=np.float32)

    pitches = idx[:, 0].astype(int)
    onsets = idx[:, 1].astype(int)

    valid = (onsets >= 0) & (onsets < pr.shape[0])
    valid_pitches = valid & (pitches >= 0) & (pitches < 88)

    pr[onsets[valid_pitches], pitches[valid_pitches]] = 5.0
    onset_channel[onsets[valid], 0] = 1.0

    pr_combined = np.hstack([pr, onset_channel])

    _pianoroll_cache[cache_key] = (pr_combined, idx)
    return pr_combined, idx


def extract_window(feat_seq: np.ndarray, t: int, win: int) -> np.ndarray:
    """Extract a fixed-size temporal window centered at frame ``t``.

    Out-of-range frames are zero-padded so that the returned window always has
    shape ``(win, num_features)``.

    Args:
        feat_seq: Feature sequence of shape ``(num_frames, num_features)``.
        t: Center frame index.
        win: Window size in frames.

    Returns:
        Float32 array of shape ``(win, num_features)``.
    """
    pad = win // 2

    t = max(0, min(t, feat_seq.shape[0] - 1))

    start = t - pad
    end = t + pad + 1 if win % 2 != 0 else t + pad

    pad_left = max(0, -start)
    pad_right = max(0, end - feat_seq.shape[0])

    safe_start = max(0, start)
    safe_end = min(feat_seq.shape[0], end)
    chunk = feat_seq[safe_start:safe_end]

    if pad_left > 0 or pad_right > 0:
        chunk = np.pad(chunk, ((pad_left, pad_right), (0, 0)), mode="constant")

    return chunk.astype(np.float32)


def onset_vector(idx: np.ndarray, T: int) -> np.ndarray:
    """Build a binary onset indicator vector of length ``T``.

    Args:
        idx: Note index array as returned by :func:`compute_pianoroll_88`.
        T: Number of frames in the target sequence.

    Returns:
        Float32 array of shape ``(T, 1)``.
    """
    v = np.zeros((T, 1), dtype=np.float32)
    onsets = idx[:, 1].astype(int)
    valid = (onsets >= 0) & (onsets < T)
    v[onsets[valid], 0] = 1.0
    return v


def resample_sequence(seq: np.ndarray, factor: float) -> np.ndarray:
    """Linearly resample a feature sequence along the temporal axis.

    Used at test time to simulate global tempo variations.

    Args:
        seq: Sequence of shape ``(T, num_features)``.
        factor: Tempo multiplier. Values below 1 expand the sequence, values
            above 1 compress it.

    Returns:
        Resampled sequence with the same dtype as ``seq``.
    """
    if abs(factor - 1.0) < 1e-6:
        return seq

    T, _ = seq.shape
    new_T = max(2, int(round(T / factor)))

    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, new_T)

    indices = np.interp(x_new, x_old, np.arange(T))
    lower = np.floor(indices).astype(int).clip(0, T - 2)
    frac = (indices - lower).reshape(-1, 1)
    out = ((1 - frac) * seq[lower] + frac * seq[lower + 1]).astype(seq.dtype)

    return out


def embed_whole_ensemble(
    feat_seq: np.ndarray,
    models: List[torch.nn.Module],
    win: int = 37,
    chunk_size: int = 512,
) -> np.ndarray:
    """Embed an entire feature sequence with an ensemble of Transformer encoders.

    Each frame is mapped to an embedding by extracting a window of size ``win``
    centered on that frame. When the ensemble members reside on distinct GPUs
    they are evaluated in parallel; otherwise they are evaluated sequentially.
    Member outputs are averaged frame by frame.

    Args:
        feat_seq: Feature sequence of shape ``(num_frames, num_features)``.
        models: Ensemble of encoders; each must implement
            :meth:`PianoTransformer.forward`.
        win: Window size used during embedding.
        chunk_size: Maximum number of windows passed to the model in a single
            batch.

    Returns:
        Mean-pooled ensemble embeddings of shape ``(num_frames, d_model)``.
    """
    pad = win // 2

    seq_padded = np.pad(feat_seq, ((pad, pad), (0, 0)), mode="constant")
    T = feat_seq.shape[0]

    # Sliding-window view via stride tricks; copied to a contiguous buffer
    # before being handed to torch.tensor.
    D = seq_padded.shape[1]
    strides = seq_padded.strides
    windows = np.lib.stride_tricks.as_strided(
        seq_padded,
        shape=(T, win, D),
        strides=(strides[0], strides[0], strides[1]),
    )
    windows = np.ascontiguousarray(windows)

    model_devices = [next(m.parameters()).device for m in models]
    distinct_devices = len({str(d) for d in model_devices}) > 1

    def _run_one_model(idx: int, m: torch.nn.Module, out_list: list) -> None:
        out_emb = []
        model_in_dim = m.input_proj.in_features
        model_device = model_devices[idx]

        with torch.no_grad():
            for i in range(0, T, chunk_size):
                batch = torch.tensor(
                    windows[i : i + chunk_size],
                    dtype=torch.float32,
                    device=model_device,
                )

                # Backward compatibility for 88-dim checkpoints.
                if model_in_dim == 88 and batch.shape[2] == 89:
                    batch = batch[:, :, :88]

                emb = m(batch).cpu().numpy()
                out_emb.append(emb)

        out_list[idx] = np.vstack(out_emb)

    all_embs: list = [None] * len(models)

    if distinct_devices:
        threads = [
            threading.Thread(target=_run_one_model, args=(idx, m, all_embs))
            for idx, m in enumerate(models)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for idx, m in enumerate(models):
            _run_one_model(idx, m, all_embs)

    return np.mean(np.stack(all_embs, axis=0), axis=0)
