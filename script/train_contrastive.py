"""Contrastive training entry point for the RAPT encoder.

Trains the :class:`PianoTransformer` as a weight-shared Siamese network with
the InfoNCE objective on aligned score--performance window pairs. Supports
single-corpus training (Vienna, Batik, (n)ASAP) and joint training across
two corpora ("asap_batik", "asap_vienna"). Held-out test pieces listed in
``test_ids_<dataset>.txt`` are removed before splitting so that the test
split used by ``run_alignment.py`` is never seen during training.

Optional augmentations applied to the performance branch only:

* temporal jitter (small frame-index shift);
* note dropout;
* local rubato simulation by index resampling;
* pitch perturbation simulating mistouches.

Resume behaviour:

* ``--resume`` re-loads an existing checkpoint at ``--model_name`` (or at
  ``--resume_from`` if given). Both the legacy "bare state_dict" format
  and the wrapped format with optimizer state and epoch counter are
  supported.
* On every epoch the script writes (i) the best-val ``state_dict`` to
  ``--model_name`` for downstream inference and (ii) a wrapped resume
  checkpoint to ``<model_name>.latest`` so future runs can continue
  without losing optimizer state.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src import architecture as models
from src import data_loader
from src import data_processor as features


def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remove_held_out_pieces(dataset: dict, test_id_file: str) -> dict:
    """Remove held-out test pieces and their variants from a dataset dict.

    Variants are identified by IDs of the form ``"<test_id>_<suffix>"`` (for
    example synthetically corrupted copies). If the file does not exist a
    warning is printed and the dataset is returned unchanged.

    Args:
        dataset: Mapping from piece id to loaded data.
        test_id_file: Path to a text file containing one held-out id per line.

    Returns:
        The dataset with held-out pieces and variants removed.
    """
    if not os.path.exists(test_id_file):
        print(f"[warning] {test_id_file} not found; nothing removed.")
        return dataset

    with open(test_id_file, "r") as f:
        test_ids = [line.strip() for line in f if line.strip()]

    removed = 0
    for tid in test_ids:
        if tid in dataset:
            dataset.pop(tid)
            removed += 1
        for v in [pid for pid in list(dataset.keys()) if pid.startswith(f"{tid}_")]:
            dataset.pop(v)
            removed += 1

    print(
        f"Removed {removed} held-out pieces (and variants) using {test_id_file}. "
        f"Remaining for train/val: {len(dataset)}"
    )
    return dataset


def load_resume_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple:
    """Load a checkpoint and restore training state.

    The function detects the checkpoint format automatically:

    * **Wrapped format** with keys ``model_state_dict``,
      ``optimizer_state_dict``, ``epoch``, ``best_val_loss`` and
      ``wait_counter`` — full state is restored and training continues from
      the saved epoch.
    * **Bare ``state_dict``** — only model weights are loaded; the optimizer
      is reinitialized and training restarts from epoch 1.

    Args:
        path: Path to the checkpoint file.
        model: Model whose weights should be restored.
        optimizer: Optimizer whose state should be restored if available.
        device: Target device for tensors loaded from the checkpoint.

    Returns:
        ``(start_epoch, best_val_loss, wait_counter)``.
    """
    if not os.path.exists(path):
        print(f"[resume] No checkpoint at {path}; starting fresh.")
        return 1, float("inf"), 0

    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[resume] Optimizer state incompatible ({e}); using fresh optimizer.")

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        wait_counter = int(ckpt.get("wait_counter", 0))
        print(
            f"[resume] Wrapped checkpoint loaded "
            f"(epoch={ckpt.get('epoch')}, best_val_loss={best_val_loss:.4f}, "
            f"wait={wait_counter}); continuing from epoch {start_epoch}."
        )
        return start_epoch, best_val_loss, wait_counter

    # Bare state_dict.
    try:
        model.load_state_dict(ckpt)
    except Exception as e:
        print(f"[resume] Strict state_dict load failed ({e}); falling back to non-strict.")
        model.load_state_dict(ckpt, strict=False)
    print("[resume] Legacy checkpoint loaded; optimizer state reset.")
    return 1, float("inf"), 0


def save_resume_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    wait_counter: int,
) -> None:
    """Write a wrapped checkpoint containing full training state."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "wait_counter": wait_counter,
        },
        path,
    )


class PairWindowDataset(Dataset):
    """Aligned (score window, performance window) pairs for contrastive training.

    For each entry ``(piece_id, score_frame, performance_frame)`` the dataset
    extracts a fixed-size window around the corresponding frame on each side
    and optionally applies augmentations to the performance window.
    """

    def __init__(
        self,
        pairs: list,
        sfeats: dict,
        pfeats: dict,
        win: int = 37,
        augment: bool = False,
        aug_dropout: float = 0.0,
        aug_stretch: bool = False,
        aug_pitch: bool = False,
        **kwargs,
    ):
        """
        Args:
            pairs: List of ``(piece_id, score_frame, performance_frame)`` tuples.
            sfeats: Mapping from piece id to score feature matrix.
            pfeats: Mapping from piece id to performance feature matrix.
            win: Window size in frames.
            augment: Whether to apply augmentations to the performance branch.
            aug_dropout: Probability of zeroing each entry in the performance
                window.
            aug_stretch: If True, simulate local rubato by jittered index
                resampling.
            aug_pitch: If True, simulate mistouches by perturbing pitches of
                a small fraction of active frames.
            **kwargs: Accepts ``in_dim`` to truncate the feature dimension
                (e.g., to drop the onset channel).
        """
        self.pairs = pairs
        self.sfeats = sfeats
        self.pfeats = pfeats
        self.win = win
        self.augment = augment
        self.aug_dropout = aug_dropout
        self.aug_stretch = aug_stretch
        self.aug_pitch = aug_pitch
        self.in_dim = kwargs.get("in_dim", 89)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple:
        pid, s_t, p_t = self.pairs[idx]

        # Temporal jitter on the performance branch.
        if self.augment and random.random() > 0.5:
            p_t = max(0, p_t + random.randint(-2, 2))

        s_win = features.extract_window(self.sfeats[pid], s_t, self.win)
        p_win = features.extract_window(self.pfeats[pid], p_t, self.win)

        s_win = s_win[:, : self.in_dim]
        p_win = p_win[:, : self.in_dim]

        if self.augment:
            if self.aug_dropout > 0:
                mask = np.random.rand(*p_win.shape) > self.aug_dropout
                p_win = p_win * mask

            if self.aug_stretch and random.random() > 0.5:
                seq_len = self.win
                indices = np.arange(seq_len, dtype=float)
                noise = np.random.normal(0, 0.6, size=seq_len)
                noise = np.convolve(noise, np.ones(3) / 3, mode="same")
                indices = np.clip(np.round(indices + noise), 0, seq_len - 1).astype(int)
                p_win = p_win[indices]

            if self.aug_pitch and random.random() > 0.5:
                pitch_dim = min(88, self.in_dim)
                active_frames, active_keys = np.where(p_win[:, :pitch_dim] > 0)
                if len(active_keys) > 0:
                    num_mistakes = max(1, int(len(active_keys) * 0.05))
                    mistake_idx = np.random.choice(
                        len(active_keys), num_mistakes, replace=False
                    )
                    for i in mistake_idx:
                        f = active_frames[i]
                        k = active_keys[i]
                        new_k = k + random.choice([-1, 1])
                        if 0 <= new_k < pitch_dim:
                            p_win[f, new_k] = p_win[f, k]
                            if random.random() > 0.5:
                                p_win[f, k] = 0

        return torch.from_numpy(s_win).float(), torch.from_numpy(p_win).float()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    temp: float = 0.07,
) -> tuple:
    """Run one training epoch with the InfoNCE objective.

    Returns:
        ``(mean_loss, mean_top1_accuracy)`` over the batches in ``loader``.
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for s_batch, p_batch in loader:
        s_batch = s_batch.to(device)
        p_batch = p_batch.to(device)

        optimizer.zero_grad()
        z_s = model(s_batch)
        z_p = model(p_batch)

        logits = torch.matmul(z_s, z_p.T) / temp
        labels = torch.arange(s_batch.size(0), device=device)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += (logits.argmax(dim=1) == labels).float().mean().item()

    return total_loss / len(loader), total_acc / len(loader)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    temp: float = 0.07,
) -> tuple:
    """Evaluate the model on a validation loader.

    Returns:
        ``(mean_loss, mean_top1_accuracy)`` over the batches in ``loader``.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for s_batch, p_batch in loader:
            s_batch = s_batch.to(device)
            p_batch = p_batch.to(device)

            z_s = model(s_batch)
            z_p = model(p_batch)

            logits = torch.matmul(z_s, z_p.T) / temp
            labels = torch.arange(s_batch.size(0), device=device)

            total_loss += criterion(logits, labels).item()
            total_acc += (logits.argmax(dim=1) == labels).float().mean().item()

    return total_loss / len(loader), total_acc / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAPT contrastive training.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="checkpoint.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--in_dim", type=int, default=89)
    parser.add_argument("--time_div", type=int, default=50)
    parser.add_argument("--win", type=int, default=37)
    parser.add_argument("--nlayers", type=int, default=3)
    parser.add_argument("--split_seed", type=int, default=1337)
    parser.add_argument("--weight_seed", type=int, default=1337)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["asap", "batik", "vienna", "combined", "asap_vienna", "asap_batik"],
        required=True,
    )
    parser.add_argument("--vienna_dir", type=str, default="./vienna4x22")
    parser.add_argument("--asap_dir", type=str, default="./asap-dataset")
    parser.add_argument("--batik_dir", type=str, default="./batik")
    parser.add_argument(
        "--pooling", type=str, default="center", choices=["center", "mean", "max"]
    )

    parser.add_argument("--aug_dropout", type=float, default=0.1)
    parser.add_argument("--aug_stretch", action="store_true")
    parser.add_argument("--aug_pitch", action="store_true")

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing checkpoint at --model_name (or "
        "--resume_from). Both legacy and wrapped checkpoint formats are "
        "accepted.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional explicit path to resume from; defaults to --model_name.",
    )

    args = parser.parse_args()
    features.TIME_DIV = args.time_div

    # 1. Load and partition the dataset(s).
    print(f"Loading dataset: {args.dataset}")

    if args.dataset == "asap_batik":
        ds_a = data_loader.load_asap_dataset(args.asap_dir)
        ds_b = data_loader.load_batik_dataset(args.batik_dir)

        ds_a = remove_held_out_pieces(ds_a, "test_ids_asap.txt")
        ds_b = remove_held_out_pieces(ds_b, "test_ids_batik.txt")

        print(f"Splitting with split_seed={args.split_seed}")
        set_seed(args.split_seed)

        t_a, v_a, s_f_a, p_f_a = data_loader.split_dataset(
            ds_a, split_seed=args.split_seed, augment=True, test_ratio=0.0
        )
        t_b, v_b, s_f_b, p_f_b = data_loader.split_dataset(
            ds_b, split_seed=args.split_seed, augment=True, test_ratio=0.0
        )

        train_pairs = t_a + t_b
        val_pairs = v_a + v_b
        s_feats = {**s_f_a, **s_f_b}
        p_feats = {**p_f_a, **p_f_b}
        print(f"Joint ASAP+Batik training pairs: {len(train_pairs)}")

    elif args.dataset == "asap_vienna":
        ds_a = data_loader.load_asap_dataset(args.asap_dir)
        ds_v = data_loader.load_vienna_dataset(args.vienna_dir)

        ds_a = remove_held_out_pieces(ds_a, "test_ids_asap.txt")
        ds_v = remove_held_out_pieces(ds_v, "test_ids_vienna.txt")

        print(f"Splitting with split_seed={args.split_seed}")
        set_seed(args.split_seed)

        t_a, v_a, s_f_a, p_f_a = data_loader.split_dataset(
            ds_a, split_seed=args.split_seed, augment=False, test_ratio=0.0
        )
        t_v, v_v, s_f_v, p_f_v = data_loader.split_dataset(
            ds_v, split_seed=args.split_seed, augment=False, test_ratio=0.0
        )

        train_pairs = t_a + t_v
        val_pairs = v_a + v_v
        s_feats = {**s_f_a, **s_f_v}
        p_feats = {**p_f_a, **p_f_v}
        print(f"Joint ASAP+Vienna training pairs: {len(train_pairs)}")

    else:
        if args.dataset == "vienna":
            dataset = data_loader.load_vienna_dataset(args.input_dir)
            test_id_file = "test_ids_vienna.txt"
        elif args.dataset == "batik":
            dataset = data_loader.load_batik_dataset(args.input_dir)
            test_id_file = "test_ids_batik.txt"
        else:
            dataset = data_loader.load_asap_dataset(args.input_dir)
            test_id_file = "test_ids_asap.txt"

        dataset = remove_held_out_pieces(dataset, test_id_file)

        print(f"Splitting with split_seed={args.split_seed}")
        set_seed(args.split_seed)

        train_pairs, val_pairs, s_feats, p_feats = data_loader.split_dataset(
            dataset, split_seed=args.split_seed, augment=True, test_ratio=0.0
        )

    # 2. Build dataloaders, model, and optimizer.
    print(f"Initializing model and augmentations with weight_seed={args.weight_seed}")
    set_seed(args.weight_seed)

    train_ds = PairWindowDataset(train_pairs, s_feats, p_feats, augment=True, **vars(args))
    val_ds = PairWindowDataset(
        val_pairs, s_feats, p_feats, augment=False, win=args.win, in_dim=args.in_dim
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = models.PianoTransformer(
        in_dim=args.in_dim, nlayers=args.nlayers, pooling=args.pooling
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    # 2b. Resume if requested. Prefer the .latest companion file (which carries
    # optimizer state) over the bare best-val checkpoint.
    start_epoch = 1
    best_val_loss = float("inf")
    wait_counter = 0
    if args.resume:
        resume_path = args.resume_from if args.resume_from else args.model_name
        latest_path = resume_path + ".latest"
        if os.path.exists(latest_path):
            resume_path = latest_path
        start_epoch, best_val_loss, wait_counter = load_resume_checkpoint(
            resume_path, model, optimizer, DEVICE
        )
        if start_epoch > args.epochs:
            print(
                f"[resume] Checkpoint already past --epochs "
                f"({start_epoch - 1} >= {args.epochs}); nothing to do."
            )
            return

    # 3. Training loop.
    print(f"Training on {DEVICE}")
    print(
        f"Epoch range: {start_epoch} to {args.epochs} "
        f"(remaining: {args.epochs - start_epoch + 1})"
    )

    latest_ckpt_path = args.model_name + ".latest"

    for epoch in range(start_epoch, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        v_loss, v_acc = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch:03d} | train loss {t_loss:.4f} acc {t_acc * 100:.1f}% | "
            f"val loss {v_loss:.4f} acc {v_acc * 100:.1f}%"
        )

        # Best-val checkpoint is saved as a bare state_dict for compatibility
        # with the inference pipeline.
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            wait_counter = 0
            torch.save(model.state_dict(), args.model_name)
            print(f"  new best val loss {best_val_loss:.4f} -> saved to {args.model_name}")
        else:
            wait_counter += 1
            print(f"  early-stopping counter: {wait_counter}/{args.patience}")

        # Wrapped resume checkpoint, written every epoch.
        save_resume_checkpoint(
            latest_ckpt_path, model, optimizer, epoch, best_val_loss, wait_counter
        )

        if wait_counter >= args.patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best validation loss: {best_val_loss:.4f}"
            )
            break

    print(f"Training finished. Best weights: {args.model_name}")
    print(f"Latest resume state: {latest_ckpt_path}")


if __name__ == "__main__":
    main()
