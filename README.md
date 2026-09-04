# NucAE

A U-Net that reconstructs dense cell-free DNA fragment centre coverage from sparse (1-5x) sequencing input, in 25 kb windows.

Setup: `conda env create -f environment.yml && conda activate nucae && pip install -e .`

Run: scripts in `scripts/` are numbered and run by hand, in order, checking each output before the next.

## The model

`nucae/unet.py` is the network, `nucae/module.py` the training recipe, and
`nucae/prewindowed.py` reads the HDF5 that produced the existing checkpoints.
Together they are the model that was actually trained on the cluster, rewritten
to be readable — not a new design.

Three variants, one class, selected by two flags:

| flags | equivalent to | parameters |
|---|---|---:|
| *(none)* | V1, and V2 with sequence | 8,517,953 |
| `--no-sequence` | V2 ablation, coverage only | 8,515,393 |
| `--dual-stream` | V3, one stream per modality | 8,545,665 |

`tests/unit/test_unet.py` checks all three against `../old_nucae/`, the code
that ran: same `state_dict` keys, old weights load with `strict=True`, and the
forward output is bit-identical. That test is why this rewrite can be trusted;
run it first.

```
pytest tests/unit                                    # 1. does the port match?
python old_nucae/inspect_h5.py --hdf5 cfdna.h5       # 2. is the data what we think?
python scripts/04_evaluate_reconstruction.py \       # 3. do the old numbers reproduce?
    --data cfdna.h5 --ckpt <existing.ckpt> --out results/gate
python scripts/03_train.py \                         # 4. only then, train
    --data cfdna.h5 --out results/v1_seed42
```

**Step 3 is a gate, not a formality.** Diff its `test_summary.txt` against the
one saved beside the checkpoint. If the mean Pearson and median peak distance do
not reproduce, something in the port is wrong and no new training result would
be interpretable. Inference is deterministic, so a difference has no seed excuse.

### What the recipe is, and is not

MSE only. Pearson r is measured on validation and carries no weight, exactly as
documented for V1. AdamW at 1e-3, weight decay 1e-2, `ReduceLROnPlateau`,
gradient clipping at 1.0.

The `grad_norm` metric is logged before clipping. Watch it: if it sits far below
1.0 the clip never fires and is free insurance, and if it sits near or above,
clipping is acting as a silent learning-rate cut and the threshold needs
revisiting.

Two things this deliberately does not do, both recorded in
`docs/OPEN_ISSUES.md`: it applies no validity mask to the loss (entry 6), and it
does not know what upstream normalisation the pre-windowed files received
(entry 7). Both matter before any number leaves this repository.
