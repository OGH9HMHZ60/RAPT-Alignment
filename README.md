# RAPT: Robust Alignment via Piano Transformer

This repository accompanies an anonymous submission and contains the code,
configuration, and evaluation entry points needed to reproduce the symbolic
score-to-performance alignment experiments reported in the paper.

> **Note on anonymity.** This repository is anonymized for double-blind
> review. Author identities, affiliations, prior work links, contact
> information, license, citation block, and acknowledgments have been
> removed and will be restored at the camera-ready stage.

---

## Overview

RAPT aligns symbolic scores to MIDI performances by combining Dynamic Time
Warping with learned contextual embeddings. A lightweight Transformer
encoder, trained contrastively on aligned score-performance pairs, produces
frame-level embeddings that are concatenated with structural pianoroll and
onset features. FastDTW operates on the resulting blended representation;
low-confidence matches are routed to a DualDTW fallback, and short gaps
between confident anchors are filled by a pitch-greedy interpolation step.

The full pipeline is described in Section RAPT of the paper. This repository
implements all components needed to train the encoder, run the inference
pipeline, and reproduce the results.

---

## Installation

A fresh Conda environment is recommended. The code is tested with
Python 3.11.

```bash
conda create -n rapt python=3.11 -y
conda activate rapt

pip install -r requirements.txt
```

The code has been tested on Linux with CUDA and on macOS with the MPS
backend. CPU-only inference is supported but slow.

---

## Datasets

Three publicly available datasets are used. Clone or download each into the
project root, matching the layout assumed by `src/data_loader.py`.

### (n)ASAP

```bash
git clone https://github.com/CPJKU/asap-dataset.git
```

The loader filters `metadata.csv` for `robust_note_alignment=True`,
yielding the 833-pair subset reported in the paper.

### Batik plays Mozart

The repository requires submodules and a one-time preprocessing pass to
generate the unfolded scores.

```bash
git clone --recurse-submodules https://github.com/huispaty/batik_plays_mozart.git
cd batik_plays_mozart
python main.py
cd ..
```

The loader expects the generated `scores_edited/` and `midi/` folders.

### Vienna 4x22

```bash
git clone https://github.com/CPJKU/vienna4x22.git
```

The loader bridges the dataset's split layout (`musicxml/`, `midi/`,
`match/`) automatically.

### Synthetic corruptions

The corrupted MIDI variants and their ground-truth alignments used in the
corrupted-test evaluation are included directly in this repository under
`data_synmist/typed_test_<dataset>/`. They were generated from the test split with the
[piano_synmist](https://github.com/Alia-morsi/piano-synmist) library using
the four error types described in the paper (`drag`,
`mistouch`, `pitch_change`, `forward_backward_insertion`). Where available,
each `.mid` file is accompanied by a `.npz` sidecar that caches the
generation-time note array; `run_alignment.py` uses the sidecar when
present and falls back to reparsing the MIDI otherwise. See
`data_synmist/README.md` for the file naming convention.

---

## Test splits

The fixed test splits used in the paper are stored at the repository root
as plain text files, one piece id per line:

```
test_ids_asap.txt
test_ids_batik.txt
test_ids_vienna.txt
```

Both training and inference scripts read these files automatically.
Training removes the listed pieces (and any synthetic variants whose ids
start with `<test_id>_`) before splitting; inference loads them as the
test set.

---

## Reproducing the paper results

Pretrained 4-seed ensembles are expected in `models/ensemble/<dataset>_3L_37w/`.
Inference reads all `.pt` files in that directory and averages their
embeddings. The per-dataset inference hyperparameters (`alpha`, `beta`,
`iou_keep`) below are the values selected on the validation split with
Optuna and reported in the supplementary hyperparameter table of the
paper.

### Clean test set

```bash
# Vienna 4x22
python -m scripts.run_alignment \
    --models ./models/ensemble/vienna_3L_37w/*.pt \
    --data_path ./vienna4x22 \
    --dataset vienna \
    --alpha 0.1369 --beta 0.7704 --iou_keep 0.8919

# Batik plays Mozart
python -m scripts.run_alignment \
    --models ./models/ensemble/batik_3L_37w/*.pt \
    --data_path ./batik_plays_mozart \
    --dataset batik \
    --alpha 0.2314 --beta 0.4987 --iou_keep 0.8315

# (n)ASAP
python -m scripts.run_alignment \
    --models ./models/ensemble/asap_3L_37w/*.pt \
    --data_path ./asap-dataset \
    --dataset asap \
    --alpha 0.8344 --beta 0.9442 --iou_keep 0.8996
```

Per-piece F1 scores are written to
`results/<dataset>_<model_folder>/f1_score_<dataset>_<model_folder>.csv`.

### Corrupted test set

Pass the directory of corrupted MIDI variants to `--corrupt_midi_dir`. The
shipped files at `data_synmist/typed_test_<dataset>/` follow the naming convention
`<piece_id>_<corruption_type>.mid` expected by the script, with a sibling
`.json` providing the ground-truth alignment.

```bash
python -m scripts.run_alignment \
    --models ./models/ensemble/vienna_3L_37w/*.pt \
    --data_path ./vienna4x22 \
    --dataset vienna \
    --corrupt_midi_dir ./data_synmist/typed_test_vienna \
    --alpha 0.1369 --beta 0.7704 --iou_keep 0.8919
```

The output CSV adds a `Corruption` column distinguishing the four error
types.

### Cross-corpus evaluation

The cross-corpus rows of Table 4 use ensembles trained on two corpora and
evaluated on the held-out third. To reproduce these, train with
`--dataset asap_batik` or `--dataset asap_vienna` (see the training
section), then run inference against the held-out corpus with the
hyperparameters selected on the joint validation split:

```bash
# ASAP+Vienna ensemble evaluated on Batik
python -m scripts.run_alignment \
    --models ./models/ensemble/asap_vienna_3L_37w/*.pt \
    --data_path ./batik_plays_mozart \
    --dataset batik \
    --alpha 0.0756 --beta 0.6951 --iou_keep 0.8515

# ASAP+Batik ensemble evaluated on Vienna
python -m scripts.run_alignment \
    --models ./models/ensemble/asap_batik_3L_37w/*.pt \
    --data_path ./vienna4x22 \
    --dataset vienna \
    --alpha 0.3319 --beta 0.8037 --iou_keep 0.8679
```

### Baselines

The three baselines reported in the paper are run with their default
configurations:

* **DualDTW**
* **Nakamura's HMM aligner**
* **GlueNote**

---

## Training from scratch

Each command trains one ensemble member. The 4-seed ensembles reported in
the paper are obtained by running the same command four times with
`--weight_seed {1001,2002,3003,4004}` and writing to four distinct `--model_name`
files. The dataset split is held fixed across seeds via `--split_seed`.

```bash
# (n)ASAP — primary training corpus
python -m scripts.train_contrastive \
    --input_dir ./asap-dataset \
    --dataset asap \
    --model_name ./models/ensemble/asap_3L_37w/seed0.pt \
    --split_seed 1337 --weight_seed 1001 \
    --epochs 200 --batch_size 64 \
    --aug_stretch --aug_pitch

# Batik plays Mozart — early stopping consistently triggers within the
# first epoch on Batik; the canonical ensemble uses a fixed 15-epoch budget.
python -m scripts.train_contrastive \
    --input_dir ./batik_plays_mozart \
    --dataset batik \
    --model_name ./models/ensemble/batik_3L_37w/seed0.pt \
    --split_seed 1337 --weight_seed 1001 \
    --epochs 15 --batch_size 16 \
    --aug_stretch --aug_pitch

# Vienna 4x22 — used primarily for the Vienna-only ensemble.
python -m scripts.train_contrastive \
    --input_dir ./vienna4x22 \
    --dataset vienna \
    --model_name ./models/ensemble/vienna_3L_37w/seed0.pt \
    --split_seed 1337 --weight_seed 1001 \
    --epochs 200 --batch_size 16 \
    --aug_stretch --aug_pitch
```

For the cross-corpus configurations, replace `--dataset`
with `asap_batik` or `asap_vienna` and provide both dataset paths via
`--asap_dir`, `--batik_dir`, or `--vienna_dir`. The `--input_dir` flag is
ignored for joint configurations but still required by the parser; any
existing path may be passed.

### Resuming a run

Training writes two checkpoints per epoch:

* the best-val `state_dict` to `--model_name`, used by inference;
* a wrapped checkpoint with optimizer state and epoch counter to
  `<model_name>.latest`, used for resuming.

To continue an interrupted run:

```bash
python -m scripts.train_contrastive \
    [same flags as before] --resume
```

The script automatically loads `<model_name>.latest` if present, falling
back to `--model_name` (a bare `state_dict`, treated as a warm start with
fresh optimizer state).

---

## Repository layout

```
.
├── src/
│   ├── architecture.py       # PianoTransformer encoder
│   ├── data_loader.py        # Loaders for ASAP, Batik, Vienna
│   ├── data_processor.py     # Pianoroll computation and window batching
│   └── dtw_core.py           # FastDTW alignment, fusion, gap interpolation
├── scripts/
│   ├── train_contrastive.py  # Contrastive training entry point
│   └── run_alignment.py      # Inference and evaluation entry point
│   
├── models/
│   └── ensemble/             # Pretrained 4-seed ensembles
│       ├── asap_3L_37w/
│       ├── batik_3L_37w/
│       ├── vienna_3L_37w/
│       ├── asap_batik_3L_37w/
│       └── asap_vienna_3L_37w/
├── data_synmist/             # Corrupted test variants and ground-truth alignments
│   ├── README.md
│   ├── typed_test_vienna/
│   ├── typed_test_batik/
│   └── typed_test_asap/
├── test_ids_asap.txt
├── test_ids_batik.txt
├── test_ids_vienna.txt
├── requirements.txt
└── README.md
```

External dataset directories (`asap-dataset/`, `batik_plays_mozart/`,
`vienna4x22/`) are expected as siblings of the project root.

---

## License

A license will be added at the camera-ready stage.
