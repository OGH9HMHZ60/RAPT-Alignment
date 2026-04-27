"""Inference script for symbolic alignment with the RAPT pipeline.

Loads a Transformer ensemble, computes a Transformer-based alignment with
optional test-time augmentation, runs DualDTW as a structural fallback,
fuses the two on a per-note basis, and writes the resulting alignment to
disk together with F1 scores when ground truth is available.
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
import parangonar as pa
import partitura as pt
import torch

from src import architecture as models
from src import data_loader
from src import data_processor as features
from src import dtw_core as alignment

warnings.filterwarnings("ignore")


def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def load_smart_models(weight_paths: list, pooling: str = "center") -> list:
    """Load a Transformer ensemble from a list of checkpoint paths.

    The input dimensionality and number of layers are inferred from each
    checkpoint's state dict. Members are distributed round-robin across
    available CUDA devices.

    Args:
        weight_paths: Paths to ``.pt`` checkpoint files.
        pooling: Pooling strategy used at training time. Must match the
            checkpoint configuration.

    Returns:
        A list of loaded, eval-mode :class:`PianoTransformer` instances.
    """
    loaded = []
    num_gpus = torch.cuda.device_count()

    for i, wp in enumerate(weight_paths):
        dev = torch.device(f"cuda:{i % num_gpus}" if num_gpus > 0 else "cpu")
        sd = torch.load(wp, map_location=dev)

        in_dim = sd["input_proj.weight"].shape[1] if "input_proj.weight" in sd else 89
        layer_keys = [k for k in sd.keys() if k.startswith("transformer.layers.")]
        n_layers = max([int(k.split(".")[2]) for k in layer_keys]) + 1 if layer_keys else 3

        print(
            f"Loading checkpoint '{wp}' on {dev} "
            f"(in_dim={in_dim}, n_layers={n_layers}, pooling={pooling})"
        )
        model = models.PianoTransformer(
            in_dim=in_dim, nlayers=n_layers, pooling=pooling
        ).to(dev)
        model.load_state_dict(sd)
        model.eval()
        loaded.append(model)

    return loaded


def _ensure_is_grace_field(score_na: np.ndarray) -> np.ndarray:
    """Add an ``is_grace`` field to ``score_na`` if it is missing."""
    if "is_grace" in score_na.dtype.names:
        return score_na
    return rfn.append_fields(
        score_na, "is_grace", np.zeros(len(score_na), dtype=bool), usemask=False
    )


def dualdtw_alignment(score_na: np.ndarray, perf_na: np.ndarray) -> list:
    """Run the parangonar DualDTW matcher and return its alignment list."""
    matcher = pa.DualDTWNoteMatcher()
    score_na = _ensure_is_grace_field(score_na)
    return matcher(score_na, perf_na, process_ornaments=False)


def fuse_transformer_with_dualdtw(
    aln_T: list,
    iou_by_score: dict,
    aln_D: list,
    iou_keep: float = 0.50,
) -> tuple:
    """Combine a Transformer alignment with a DualDTW alignment.

    A Transformer match is kept whenever its IoU is at least ``iou_keep``;
    otherwise the DualDTW match for the same score note is used, provided the
    target performance note has not been claimed already. Score notes that
    remain unmatched are recorded as deletions.

    Args:
        aln_T: Transformer alignment.
        iou_by_score: Per-score-note IoU confidence from the Transformer pass.
        aln_D: DualDTW alignment.
        iou_keep: Minimum IoU required to keep a Transformer match.

    Returns:
        ``(fused_alignment, used_perf)``, where ``fused_alignment`` covers the
        score side only and ``used_perf`` is the set of claimed performance
        ids.
    """
    dual_match_by_score = {
        str(e["score_id"]): str(e["performance_id"])
        for e in aln_D
        if e.get("label") == "match"
    }

    used_perf = set()
    fused = []

    for event in aln_T:
        if event.get("label") == "insertion":
            continue

        sid = str(event["score_id"])
        t_iou = float(iou_by_score.get(sid, 0.0))

        if event["label"] == "match" and t_iou >= iou_keep:
            pid = str(event["performance_id"])
            fused.append({"label": "match", "score_id": sid, "performance_id": pid})
            used_perf.add(pid)
            continue

        if sid in dual_match_by_score:
            pid = dual_match_by_score[sid]
            if pid not in used_perf:
                fused.append(
                    {"label": "match", "score_id": sid, "performance_id": pid}
                )
                used_perf.add(pid)
                continue

        fused.append({"label": "deletion", "score_id": sid})

    return fused, used_perf


def add_insertions(
    fused_score_side: list,
    used_perf: set,
    perf_na: np.ndarray,
) -> list:
    """Append insertion entries for performance notes that remain unclaimed."""
    final = list(fused_score_side)
    for pn in perf_na:
        pid = str(pn["id"])
        if pid not in used_perf:
            final.append({"label": "insertion", "performance_id": pid})
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="RAPT inference pipeline.")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Paths to Transformer checkpoint weights (.pt).")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="vienna",
                        choices=["asap", "batik", "vienna"])
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--save_f1", type=str, default=None)
    parser.add_argument("--tta", type=float, nargs="+", default=[1.0])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--iou_keep", type=float, default=0.50)
    parser.add_argument("--win", type=int, default=37)
    parser.add_argument("--time_div", type=int, default=50)
    parser.add_argument("--pooling", type=str, default="center",
                        choices=["center", "mean", "max"],
                        help="Pooling strategy used at training time; must "
                             "match the checkpoint configuration.")
    parser.add_argument("--eval_all", action="store_true")
    parser.add_argument("--corrupt_midi_dir", type=str, default=None)
    parser.add_argument("--disable_interpolation", action="store_true")
    args = parser.parse_args()

    features.TIME_DIV = args.time_div

    # Resolve default output paths from the model directory name.
    if args.models:
        first_model_dir = os.path.dirname(args.models[0])
        model_folder_name = (
            os.path.basename(first_model_dir) if first_model_dir else "unknown_model"
        )
    else:
        model_folder_name = "unknown_model"

    if args.out_dir is None:
        args.out_dir = os.path.join(
            "results", f"{args.dataset}_{model_folder_name}"
        )
    if args.save_f1 is None:
        args.save_f1 = os.path.join(
            args.out_dir, f"f1_score_{args.dataset}_{model_folder_name}.csv"
        )
    elif args.save_f1.upper() == "DISABLE":
        args.save_f1 = None

    os.makedirs(args.out_dir, exist_ok=True)
    ensemble = load_smart_models(args.models, pooling=args.pooling)

    print(f"Loading {args.dataset.upper()} dataset...")
    if args.dataset == "vienna":
        ds_raw = data_loader.load_vienna_dataset(args.data_path)
    elif args.dataset == "batik":
        ds_raw = data_loader.load_batik_dataset(args.data_path)
    else:
        ds_raw = data_loader.load_asap_dataset(args.data_path)

    test_id_file = f"test_ids_{args.dataset}.txt"
    if os.path.exists(test_id_file):
        print(f"Using fixed test split from {test_id_file}")
        with open(test_id_file, "r") as f:
            test_ids = [line.strip() for line in f if line.strip()]
        test_ds = {k: ds_raw[k] for k in test_ids if k in ds_raw}
    else:
        print(f"[warning] {test_id_file} not found; using a dynamic split.")
        _, _, test_ds = data_loader.create_splits(ds_raw, seed=args.seed)

    eval_ds = ds_raw if args.eval_all else test_ds

    print(f"Ensemble size: {len(ensemble)}")
    print(f"Output directory: {args.out_dir}")
    print(f"Pieces to evaluate: {len(eval_ds)}")

    f1_scores = []
    f1_records = []

    for piece_id, (perf_na, score_na, gt) in eval_ds.items():
        if args.corrupt_midi_dir is not None:
            pattern = os.path.join(args.corrupt_midi_dir, f"{piece_id}_*.mid")
            corrupted_files = glob.glob(pattern)
            if not corrupted_files:
                print(f"[warning] No corrupted variants for {piece_id}; skipping.")
                continue

            for corrupted_midi_path in corrupted_files:
                corruption_type = (
                    os.path.basename(corrupted_midi_path)
                    .replace(f"{piece_id}_", "")
                    .replace(".mid", "")
                )
                try:
                    corrupted_npz_path = corrupted_midi_path.replace(".mid", ".npz")
                    if os.path.exists(corrupted_npz_path):
                        npz_data = np.load(corrupted_npz_path, allow_pickle=True)
                        perf_na_corrupt = npz_data["note_array"]
                    else:
                        perf = pt.load_performance(corrupted_midi_path)
                        perf_na_corrupt = perf.note_array(include_pitch_spelling=True)

                    corrupted_align_path = corrupted_midi_path.replace(".mid", ".json")
                    if os.path.exists(corrupted_align_path):
                        with open(corrupted_align_path, "r") as f:
                            gt_alignment_eval = json.load(f)
                    else:
                        gt_alignment_eval = gt
                except Exception as e:
                    print(f"[error] Failed to load {corrupted_midi_path}: {e}")
                    continue

                aln_T, iou_T = alignment.tta_transformer_alignment(
                    score_na, perf_na_corrupt, ensemble,
                    factors=args.tta, alpha=args.alpha,
                    beta=args.beta, win=args.win,
                )
                aln_D = dualdtw_alignment(score_na, perf_na_corrupt)
                fused_score_side, used_perf = fuse_transformer_with_dualdtw(
                    aln_T, iou_T, aln_D, iou_keep=args.iou_keep
                )
                fused_pre_gap = add_insertions(
                    fused_score_side, used_perf, perf_na_corrupt
                )

                _, s_idx = features.compute_pianoroll_88(score_na)
                _, p_idx = features.compute_pianoroll_88(perf_na_corrupt)
                final_aln = alignment.fill_gaps_interpolation(
                    fused_pre_gap, s_idx, p_idx, score_na, perf_na_corrupt
                )

                out_name = f"{piece_id}_{corruption_type}.csv"
                pd.DataFrame(final_aln).to_csv(
                    os.path.join(args.out_dir, out_name), index=False
                )

                if gt:
                    _, _, f_score = data_loader.compare_alignments(
                        final_aln, gt_alignment_eval
                    )
                    print(f"[{piece_id} | {corruption_type}] F1 = {f_score:.4f}")
                    f1_records.append(
                        {"Piece": piece_id, "Corruption": corruption_type, "F1": f_score}
                    )

        else:
            aln_T, iou_T = alignment.tta_transformer_alignment(
                score_na, perf_na, ensemble, factors=args.tta,
                alpha=args.alpha, beta=args.beta, win=args.win,
            )
            aln_D = dualdtw_alignment(score_na, perf_na)
            fused_score_side, used_perf = fuse_transformer_with_dualdtw(
                aln_T, iou_T, aln_D, iou_keep=args.iou_keep
            )
            fused_pre_gap = add_insertions(fused_score_side, used_perf, perf_na)

            if args.disable_interpolation:
                final_aln = fused_pre_gap
            else:
                _, s_idx = features.compute_pianoroll_88(score_na)
                _, p_idx = features.compute_pianoroll_88(perf_na)
                final_aln = alignment.fill_gaps_interpolation(
                    fused_pre_gap, s_idx, p_idx, score_na, perf_na
                )

            pd.DataFrame(final_aln).to_csv(
                os.path.join(args.out_dir, f"{piece_id}.csv"), index=False
            )

            if gt:
                _, _, f_score = data_loader.compare_alignments(final_aln, gt)
                print(f"[{piece_id}] F1 = {f_score:.4f}")
                f1_scores.append(f_score)
                f1_records.append(
                    {"Piece": piece_id, "Corruption": "clean", "F1": f_score}
                )

    if args.save_f1 and f1_records:
        pd.DataFrame(f1_records).to_csv(args.save_f1, index=False)
        print(f"\nF1 scores written to {args.save_f1}")

    if f1_scores:
        avg_f1 = sum(f1_scores) / len(f1_scores)
        print(f"\nMean F1 over {len(f1_scores)} pieces: {avg_f1:.4f}")


if __name__ == "__main__":
    main()
