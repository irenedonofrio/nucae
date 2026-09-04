"""The training recipe does what it says: MSE loss, the schedule that ran,
a Pearson that agrees with numpy, and a gradient norm logged before clipping.
"""

from __future__ import annotations

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nucae.metrics import per_window_pearson
from nucae.module import MIN_LR, NucAEModule

LENGTH = 256         # divisible by 8 and above the 192 bp reflect-padding floor
SEED = 0


def tiny_loaders(n: int = 4, batch: int = 2) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(SEED)
    x = torch.randn(n, 6, LENGTH, generator=generator)
    y = torch.randn(n, 1, LENGTH, generator=generator)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch), DataLoader(dataset, batch_size=batch)


def tiny_trainer(**kwargs) -> pl.Trainer:
    return pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False, **kwargs)


# ── the objective ─────────────────────────────────────────────────────────────
def test_training_loss_is_plain_mse():
    """No Pearson term, no alpha. The documented V1 objective."""
    torch.manual_seed(SEED)
    module = NucAEModule()
    x, y = next(iter(tiny_loaders()[0]))
    with torch.no_grad():
        assert torch.equal(module.training_step((x, y), 0), F.mse_loss(module(x), y))


# ── the metric ────────────────────────────────────────────────────────────────
def test_pearson_matches_numpy():
    generator = torch.Generator().manual_seed(SEED)
    a = torch.randn(3, 1, 500, generator=generator)
    b = torch.randn(3, 1, 500, generator=generator)
    expected = [np.corrcoef(a[i, 0].numpy(), b[i, 0].numpy())[0, 1] for i in range(3)]
    assert np.allclose(per_window_pearson(a, b).numpy(), expected, atol=1e-6)


def test_pearson_on_a_flat_window_is_finite_and_near_zero():
    """A blacklisted window is constant. It must score badly, not produce NaN."""
    flat = torch.zeros(1, 1, 500)
    other = torch.randn(1, 1, 500, generator=torch.Generator().manual_seed(SEED))
    r = per_window_pearson(flat, other)
    assert torch.isfinite(r).all() and abs(float(r)) < 1e-6


def test_pearson_rejects_a_missing_channel_axis():
    """(B, L) would broadcast into (B, B, L) and return a plausible wrong number."""
    with pytest.raises(ValueError):
        per_window_pearson(torch.randn(2, 500), torch.randn(2, 500))


# ── the schedule ──────────────────────────────────────────────────────────────
def test_optimiser_and_schedule_match_what_ran():
    config = NucAEModule(lr=1e-3, weight_decay=1e-2, lr_patience=7).configure_optimizers()
    optimizer = config["optimizer"]
    scheduler = config["lr_scheduler"]["scheduler"]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 1e-2
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert (scheduler.factor, scheduler.patience) == (0.5, 7)
    assert scheduler.min_lrs == [MIN_LR]
    assert config["lr_scheduler"]["monitor"] == "val_loss"


# ── provenance ────────────────────────────────────────────────────────────────
def test_hyperparameters_survive_a_checkpoint_round_trip(tmp_path):
    """The recipe must be recoverable from its own .ckpt, unlike the baseline."""
    torch.manual_seed(SEED)
    module = NucAEModule(lr=3e-4, use_sequence=False, lr_patience=3)
    trainer = tiny_trainer(max_steps=1, limit_val_batches=0)
    train, _ = tiny_loaders()
    x = torch.randn(4, 1, LENGTH, generator=torch.Generator().manual_seed(SEED))
    y = torch.randn(4, 1, LENGTH, generator=torch.Generator().manual_seed(SEED))
    trainer.fit(module, DataLoader(TensorDataset(x, y), batch_size=2))

    path = tmp_path / "round_trip.ckpt"
    trainer.save_checkpoint(path)
    restored = NucAEModule.load_from_checkpoint(path, map_location="cpu")

    assert restored.hparams.lr == 3e-4
    assert restored.hparams.use_sequence is False
    assert restored.hparams.lr_patience == 3
    for key, tensor in module.state_dict().items():
        assert torch.equal(restored.state_dict()[key], tensor)


def test_checkpoint_keys_carry_the_model_prefix():
    """Old cluster checkpoints are 'model.'-prefixed; renaming the attribute
    would make load_from_checkpoint fail on every one of them."""
    keys = NucAEModule().state_dict()
    assert all(k.startswith("model.") for k in keys)


# ── clipping diagnostics ──────────────────────────────────────────────────────
def test_logged_gradient_norm_is_measured_before_clipping():
    """Pins the hook ordering the diagnostic depends on.

    With the clip threshold at 1e-8, a post-clip norm could not exceed it. A
    logged value far above it therefore proves the number is the pre-clip norm,
    which is the only one that says whether clipping is firing.
    """
    torch.manual_seed(SEED)
    module = NucAEModule()
    train, val = tiny_loaders()
    trainer = tiny_trainer(max_epochs=1, gradient_clip_val=1e-8)
    trainer.fit(module, train, val)

    logged = float(trainer.logged_metrics["grad_norm"])
    assert logged > 1e-6
