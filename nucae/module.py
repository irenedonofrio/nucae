"""The training recipe: MSE, AdamW, ReduceLROnPlateau. What actually ran.

This is the documented V1 recipe -- MSE only, with Pearson r measured on the
validation set but carrying no weight in the loss. The composite
`(1 - r) + alpha * MSE` objective is deliberately NOT here: alpha scales as
1/var(signal), so it is a property of the dataset rather than of the objective,
and it must be measured on real windows before it can be chosen. That is the
next experiment, not this baseline.

LOADING EXISTING CLUSTER CHECKPOINTS. The attribute is `self.model`, not `net`,
and the saved hyperparameters are the same names the cluster module saved, so

    NucAEModule.load_from_checkpoint(path)

works directly on the existing .ckpt files. One exception: a V3 checkpoint was
written by a module that had no `dual_stream` hyperparameter and built the
two-stream network unconditionally, so it must be loaded as

    NucAEModule.load_from_checkpoint(path, dual_stream=True)

Without it the state_dict load fails loudly on unexpected `coverage_stream.*`
keys, which is the right failure -- it cannot silently load the wrong network.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from nucae.metrics import per_window_pearson
from nucae.unet import FragmentomicsUNet

# The cluster code used patience=7; the thesis draft says 3. 7 is what ran, so it
# is the default here. At 40-50 epochs the two behave very differently -- 7 will
# rarely fire -- so this is a real choice and not a formality.
LR_PATIENCE = 7
LR_FACTOR = 0.5
MIN_LR = 1e-5


class NucAEModule(pl.LightningModule):
    """Wraps FragmentomicsUNet with the loss, optimiser and schedule that ran."""

    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-2,
                 use_sequence: bool = True, dual_stream: bool = False,
                 lr_patience: int = LR_PATIENCE,
                 masked_loss: bool = False) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = FragmentomicsUNet(use_sequence=use_sequence,
                                       dual_stream=dual_stream)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def _unpack(batch):
        """(x, y) from the pre-windowed format, (x, y, valid) from the counts one.

        The third element is a LOSS WEIGHT, and only nucae.data.WindowDataset
        can supply it: the pre-windowed files carry no mask at all. Returning
        None rather than a tensor of ones keeps that distinction visible -- "no
        mask exists" and "every position is valid" are different claims, and
        the second one is not something this repository can make about the
        inherited files.
        """
        return batch if len(batch) == 3 else (batch[0], batch[1], None)

    def _loss(self, y_hat: torch.Tensor, y: torch.Tensor,
              valid: torch.Tensor | None) -> torch.Tensor:
        """MSE, weighted by validity only when BOTH a mask exists and it is asked for.

        `masked_loss` defaults False so that a run on counts-derived data
        changes the DATA and nothing else. Turning it on is a change to the
        objective and therefore a separate experiment with its own number --
        moving both at once would leave neither attributable.

        Weighted, the denominator is the count of valid positions, not the
        window length: dividing by the length would shrink the loss of a
        heavily masked window toward zero and quietly teach the model that
        unmappable regions are already solved.
        """
        if not self.hparams.masked_loss or valid is None:
            return F.mse_loss(y_hat, y)
        weight = valid.unsqueeze(1)                     # (B, L) -> (B, 1, L)
        return (((y_hat - y) ** 2) * weight).sum() / weight.sum().clamp(min=1.0)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y, valid = self._unpack(batch)
        loss = self._loss(self(x), y, valid)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y, valid = self._unpack(batch)
        y_hat = self(x)
        loss = self._loss(y_hat, y, valid)
        self.log("val_loss", loss, prog_bar=True)
        # Measured, not optimised. Reported so the shape/amplitude split is
        # visible: MSE can fall while r stalls, and that is worth seeing early.
        self.log("val_pearson_r", per_window_pearson(y_hat, y).mean(), prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log the global gradient norm BEFORE clipping.

        Without this there is no way to tell whether gradient_clip_val=1.0 is
        inert insurance or is firing on most steps -- in which case it is acting
        as a silent learning-rate cut rather than a safety net.

        The L2 norm of the per-parameter L2 norms is the global L2 norm, and
        avoids materialising a copy of every gradient.
        """
        norms = [p.grad.detach().norm() for p in self.parameters() if p.grad is not None]
        if norms:
            self.log("grad_norm", torch.stack(norms).norm())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_FACTOR,
            patience=self.hparams.lr_patience, min_lr=MIN_LR)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss",
                             "interval": "epoch", "frequency": 1},
        }
