"""Alignment algorithms based on Dynamic Time Warping.

Combines Transformer embeddings with structural pianoroll features through a
weighted concatenation, computes a frame-level warping path with FastDTW, and
resolves it into a discrete note-to-note alignment via an Intersection-over-
Union (IoU) heuristic. Also provides a test-time augmentation wrapper and a
post-hoc gap-filling routine.
"""

from typing import List, Tuple

import numpy as np
import torch
from fastdtw import fastdtw

from src import data_processor as features


def greedy_note_alignment_iou_with_conf(
    warping_path: np.ndarray,
    idx_s: np.ndarray,
    score_na: np.ndarray,
    idx_p: np.ndarray,
    perf_na: np.ndarray,
    min_iou: float = 0.05,
) -> Tuple[list, dict]:
    """Resolve a frame-level DTW path into a note-level alignment.

    For every score note the function selects, among performance notes of the
    same pitch, the one whose frame interval has the highest temporal IoU with
    the projected interval implied by the warping path. Each performance note
    can be assigned at most once.

    Args:
        warping_path: ``(N, 2)`` array of (score_frame, perf_frame) pairs.
        idx_s: Score note frame bounds as returned by Partitura.
        score_na: Structured score note array.
        idx_p: Performance note frame bounds as returned by Partitura.
        perf_na: Structured performance note array.
        min_iou: Minimum IoU below which a candidate match is rejected.

    Returns:
        A tuple ``(alignment, iou_by_score)``. ``alignment`` is a list of
        match/deletion entries; ``iou_by_score`` maps each score id to the IoU
        of its best candidate (used downstream as a confidence score).
    """
    used_perf = set()
    alignment = []
    iou_by_score = {}

    perf_by_pitch: dict = {}
    for j, pn in enumerate(perf_na):
        perf_by_pitch.setdefault(int(pn["pitch"]), []).append(j)

    warping_path = np.asarray(warping_path)

    for i, sn in enumerate(score_na):
        sid = str(sn["id"])
        pitch = int(sn["pitch"])
        s1 = int(idx_s[i, 1])
        e1 = int(idx_s[i, 2])

        mask = (warping_path[:, 0] >= s1) & (warping_path[:, 0] <= e1)
        if not np.any(mask):
            alignment.append({"label": "deletion", "score_id": sid})
            iou_by_score[sid] = 0.0
            continue

        mn = int(warping_path[mask, 1].min())
        mx = int(warping_path[mask, 1].max())

        best_pid = None
        best_iou = 0.0

        for j in perf_by_pitch.get(pitch, []):
            pid = str(perf_na[j]["id"])
            if pid in used_perf:
                continue

            s2 = int(idx_p[j, 1])
            e2 = int(idx_p[j, 2])

            inter = max(0, min(mx, e2) - max(mn, s2) + 1)
            if inter == 0:
                continue

            union = (mx - mn + 1) + (e2 - s2 + 1) - inter
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_pid = pid

        iou_by_score[sid] = best_iou

        if best_pid is not None and best_iou >= min_iou:
            alignment.append(
                {"label": "match", "score_id": sid, "performance_id": best_pid}
            )
            used_perf.add(best_pid)
        else:
            alignment.append({"label": "deletion", "score_id": sid})

    # Insertions are handled outside this function.
    return alignment, iou_by_score


def transformer_alignment(
    score_na: np.ndarray,
    perf_na: np.ndarray,
    models: List[torch.nn.Module],
    device: torch.device,
    win: int = 37,
    alpha: float = 0.6,
    beta: float = 0.4,
    dtw_radius: int = 50,
    stretch_factor: float = 1.0,
) -> Tuple[list, dict]:
    """Align two note arrays using FastDTW on blended features.

    Each side is converted to a frame-level feature sequence by concatenating
    Transformer embeddings (weight ``alpha``) with the normalized pianoroll and
    the onset indicator (each weighted by ``beta``). FastDTW computes the
    warping path, which is then resolved into a note-level alignment.

    Args:
        score_na: Structured score note array.
        perf_na: Structured performance note array.
        models: Transformer ensemble used to produce embeddings.
        device: Compute device (kept for API symmetry; per-model devices are
            handled internally by :func:`embed_whole_ensemble`).
        win: Window size for embedding extraction.
        alpha: Weight applied to Transformer embeddings before concatenation.
        beta: Weight applied to pianoroll and onset features before
            concatenation.
        dtw_radius: FastDTW search radius (Sakoe-Chiba band).
        stretch_factor: Tempo multiplier applied to the performance side.

    Returns:
        ``(alignment, iou_by_score)`` as in
        :func:`greedy_note_alignment_iou_with_conf`.
    """
    s_pr, s_idx = features.compute_pianoroll_88(score_na)
    p_pr, p_idx = features.compute_pianoroll_88(perf_na)

    if stretch_factor != 1.0:
        p_pr = features.resample_sequence(p_pr, stretch_factor)
        p_idx = np.round(p_idx * stretch_factor).astype(int)

    E_s = features.embed_whole_ensemble(s_pr, models, win=win)
    E_p = features.embed_whole_ensemble(p_pr, models, win=win)

    s_on = features.onset_vector(s_idx, s_pr.shape[0])
    p_on = features.onset_vector(p_idx, p_pr.shape[0])

    s_pr_norm = s_pr / 5.0
    p_pr_norm = p_pr / 5.0

    S_blend = np.hstack([alpha * E_s, beta * s_pr_norm, beta * s_on])
    P_blend = np.hstack([alpha * E_p, beta * p_pr_norm, beta * p_on])

    _, path = fastdtw(S_blend, P_blend, dist=2, radius=dtw_radius)
    path = np.asarray(path)

    return greedy_note_alignment_iou_with_conf(
        path, s_idx, score_na, p_idx, perf_na, min_iou=0.05
    )


def tta_transformer_alignment(
    score_na: np.ndarray,
    perf_na: np.ndarray,
    models: List[torch.nn.Module],
    factors: List[float] = [1.0],
    alpha: float = 0.5,
    beta: float = 0.5,
    win: int = 37,
) -> Tuple[list, dict]:
    """Align with test-time augmentation across a set of tempo factors.

    The base alignment is computed once for each factor in ``factors``. For
    each score note, every tempo-stretched run contributes a vote for one
    performance note; ties are broken by mean IoU. The resulting consensus
    alignment is returned together with per-note confidence scores.

    Args:
        score_na: Structured score note array.
        perf_na: Structured performance note array.
        models: Transformer ensemble.
        factors: Tempo multipliers to evaluate.
        alpha: Weight for Transformer embeddings.
        beta: Weight for pianoroll and onset features.
        win: Window size for embedding extraction.

    Returns:
        ``(alignment, iou_by_score)`` analogous to
        :func:`transformer_alignment`.
    """
    device = next(models[0].parameters()).device

    cand: dict = {}

    for f in factors:
        aln_T_f, iou_f = transformer_alignment(
            score_na, perf_na, models, device,
            win=win, alpha=alpha, beta=beta, stretch_factor=f,
        )
        for e in aln_T_f:
            if e["label"] == "match":
                sid = str(e["score_id"])
                pid = str(e["performance_id"])
                cand.setdefault(sid, []).append((pid, float(iou_f.get(sid, 0.0))))

    fused = []
    used_perf = set()
    iou_by_score = {}

    for sn in score_na:
        sid = str(sn["id"])
        options = cand.get(sid, [])

        if not options:
            fused.append({"label": "deletion", "score_id": sid})
            iou_by_score[sid] = 0.0
            continue

        freq: dict = {}
        for pid, iou in options:
            freq.setdefault(pid, []).append(iou)

        best_pid, best_votes, best_avg_iou = None, -1, -1.0
        for pid, ious in freq.items():
            votes = len(ious)
            avg_iou = float(np.mean(ious))
            if votes > best_votes or (votes == best_votes and avg_iou > best_avg_iou):
                best_pid, best_votes, best_avg_iou = pid, votes, avg_iou

        if best_pid is not None and best_pid not in used_perf:
            fused.append(
                {"label": "match", "score_id": sid, "performance_id": best_pid}
            )
            used_perf.add(best_pid)
            iou_by_score[sid] = best_avg_iou
        else:
            fused.append({"label": "deletion", "score_id": sid})
            iou_by_score[sid] = 0.0

    return fused, iou_by_score


def fill_gaps_interpolation(
    pred_alignment: list,
    s_idx: np.ndarray,
    p_idx: np.ndarray,
    score_na: np.ndarray,
    perf_na: np.ndarray,
    max_gap_seconds: float = 1.5,
) -> list:
    """Fill short gaps between confident matches with pitch-greedy assignments.

    The function scans pairs of consecutive confirmed matches and, when the
    intermediate score region is short enough in performance time, assigns each
    unmatched score note to the next free performance note of the same pitch.
    Score deletions that are recovered this way are removed from the result.

    Args:
        pred_alignment: Alignment produced by an upstream method.
        s_idx: Score note frame bounds.
        p_idx: Performance note frame bounds.
        score_na: Structured score note array.
        perf_na: Structured performance note array.
        max_gap_seconds: Maximum allowed gap, in seconds, between two anchor
            matches for interpolation to be attempted.

    Returns:
        The augmented alignment.
    """
    s_map = {str(n["id"]): i for i, n in enumerate(score_na)}
    p_map = {str(n["id"]): i for i, n in enumerate(perf_na)}

    matches = [x for x in pred_alignment if x["label"] == "match"]
    matches = [
        m for m in matches
        if str(m["score_id"]) in s_map and str(m["performance_id"]) in p_map
    ]
    matches.sort(key=lambda x: s_map[str(x["score_id"])])

    used_perf = {str(x["performance_id"]) for x in matches}
    new_matches = []

    fps = features.TIME_DIV
    max_gap_frames = int(max_gap_seconds * fps)

    for k in range(len(matches) - 1):
        m1, m2 = matches[k], matches[k + 1]
        s1_idx, s2_idx = s_map[str(m1["score_id"])], s_map[str(m2["score_id"])]

        if s2_idx <= s1_idx + 1:
            continue

        p1_idx, p2_idx = p_map[str(m1["performance_id"])], p_map[str(m2["performance_id"])]
        if p2_idx <= p1_idx:
            continue

        t_start = int(p_idx[p1_idx, 2])
        t_end = int(p_idx[p2_idx, 1])
        if (t_end - t_start) > max_gap_frames:
            continue

        for curr_s_idx in range(s1_idx + 1, s2_idx):
            curr_s_note = score_na[curr_s_idx]
            curr_pitch = int(curr_s_note["pitch"])
            sid = str(curr_s_note["id"])

            for cand_p_idx in range(p1_idx + 1, p2_idx):
                pid = str(perf_na[cand_p_idx]["id"])
                if pid in used_perf:
                    continue
                if int(perf_na[cand_p_idx]["pitch"]) == curr_pitch:
                    new_matches.append(
                        {"label": "match", "score_id": sid, "performance_id": pid}
                    )
                    used_perf.add(pid)
                    break

    recovered_ids = {x["score_id"] for x in new_matches}
    final_aln = [
        x for x in pred_alignment
        if not (x["label"] == "deletion" and str(x["score_id"]) in recovered_ids)
    ]
    final_aln.extend(new_matches)
    return final_aln
