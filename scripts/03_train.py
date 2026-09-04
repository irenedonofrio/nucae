#!/usr/bin/env python
"""Train the U-Net on a pre-windowed HDF5. One run, one output directory.

Defaults reproduce the recipe that produced the existing cluster checkpoints:
MSE loss, AdamW at 1e-3 with weight decay 1e-2, ReduceLROnPlateau, gradient
clipping at 1.0. Pearson r is logged on validation but carries no weight.

Variants (see nucae/unet.py): no flags is V1 / V2-with-sequence, --no-sequence
is the V2 ablation, --dual-stream is V3.

    python scripts/03_train.py --data cfdna_fullgenome.h5 --out results/v1_seed42
"""
import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from nucae.module import LR_PATIENCE, NucAEModule
from nucae.prewindowed import CfDNAWindowsDataModule

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", type=Path, required=True, help="pre-windowed cfdna_*.h5")
ap.add_argument("--out", type=Path, required=True, help="run directory; created")
ap.add_argument("--no-sequence", action="store_true", help="V2 ablation: coverage only")
ap.add_argument("--dual-stream", action="store_true", help="V3: one stream per modality")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--weight-decay", type=float, default=1e-2)
ap.add_argument("--lr-patience", type=int, default=LR_PATIENCE,
                help="epochs without val_loss improvement before the LR is halved")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--workers", type=int, default=4, help="match --cpus-per-task")
ap.add_argument("--seed", type=int, default=42, help="recorded in run_config.json")
ap.add_argument("--resume", type=Path, default=None,
                help="checkpoint to continue from, e.g. <out>/checkpoints/last.ckpt. "
                     "Restores optimiser and scheduler state as well as weights, so a "
                     "job killed at the walltime resumes rather than starting over.")
a = ap.parse_args()

# Before anything is built, so weight initialisation is covered as well as data
# order. workers=True re-seeds each DataLoader worker; without it every worker
# inherits one RNG state and draws the same sequence.
pl.seed_everything(a.seed, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

a.out.mkdir(parents=True, exist_ok=True)
(a.out / "run_config.json").write_text(
    json.dumps({**{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                "torch": torch.__version__, "lightning": pl.__version__}, indent=2))

data = CfDNAWindowsDataModule(a.data, batch_size=a.batch_size, num_workers=a.workers,
                              use_sequence=not a.no_sequence)
model = NucAEModule(lr=a.lr, weight_decay=a.weight_decay,
                    use_sequence=not a.no_sequence, dual_stream=a.dual_stream,
                    lr_patience=a.lr_patience)

trainer = pl.Trainer(
    max_epochs=a.epochs,
    default_root_dir=str(a.out),
    logger=CSVLogger(save_dir=str(a.out), name="logs"),
    gradient_clip_val=1.0,      # inert unless grad_norm approaches it; watch that metric
    log_every_n_steps=10,
    callbacks=[
        ModelCheckpoint(dirpath=str(a.out / "checkpoints"), monitor="val_loss",
                        mode="min", save_top_k=3, save_last=True,
                        filename="epoch={epoch:02d}-val_loss={val_loss:.5f}",
                        auto_insert_metric_name=False),
        LearningRateMonitor(logging_interval="epoch"),
    ],
)
trainer.fit(model, datamodule=data, ckpt_path=a.resume)
print(f"done: {a.out}")
