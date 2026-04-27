# Corrupted test variants

This directory contains the corrupted MIDI files and ground-truth
alignments used in the corrupted-test evaluation reported in Section 5.2
of the paper. They are included so that the corrupted-test numbers can be
reproduced directly, without rerunning the generation step.

## Layout

```
data_synmist/
├── vienna/
│   ├── <piece_id>_drag.mid
│   ├── <piece_id>_drag.json
│   ├── <piece_id>_mistouch.mid
│   ├── <piece_id>_mistouch.json
│   ├── <piece_id>_pitch_change.mid
│   ├── <piece_id>_pitch_change.json
│   ├── <piece_id>_forward_backward_insertion.mid
│   └── <piece_id>_forward_backward_insertion.json
├── batik/
└── asap/
```

For every test piece in `test_ids_<dataset>.txt`, four corrupted variants
are provided, one per mistake type (`drag`, `mistouch`, `pitch_change`,
`forward_backward_insertion`). The naming convention
`<piece_id>_<mistake_type>.mid` matches what `scripts/run_alignment.py`
expects via the `--corrupt_midi_dir` flag. Each `.mid` file is paired with
a `.json` file containing the ground-truth alignment in the namespace of
that MIDI's note ids.

Some variants additionally ship a `.npz` sidecar file with the same stem.
The sidecar caches the note array as produced at generation time;
`run_alignment.py` loads it in preference to reparsing the MIDI when
present, which avoids minor differences introduced by partitura's MIDI
reload. The sidecar is optional: if a reviewer regenerates the variants or
the `.npz` is missing, evaluation falls back to loading the MIDI directly.

## How these were generated

The files were produced from the clean test performances using the
[piano_synmist](https://github.com/Alia-morsi/piano-synmist) library, with
20 mistakes per type per piece. Ground-truth alignments are constructed by
matching clean-performance notes to their corrupted counterparts via
`(onset_sec, pitch)`, which is robust to partitura's note-id reassignment
on MIDI reload.

## Use

To evaluate on the corrupted variants:

```bash
python -m scripts.run_alignment \
    --models ./checkpoints/<dataset>/*.pt \
    --data_path <path to clean dataset> \
    --dataset <vienna|batik|asap> \
    --corrupt_midi_dir data_synmist/<dataset>
```

`run_alignment.py` automatically picks up the `.json` ground-truth files
sitting next to each corrupted `.mid`, and writes per-corruption F1 scores
to the output CSV.
