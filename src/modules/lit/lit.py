# src/modules/lit_ccs/lit_ccs.py
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from schedulefree import RAdamScheduleFree
from torchmetrics import PearsonCorrCoef


class LitMultiTask(pl.LightningModule):
    def __init__(
        self,
        core: torch.nn.Module,
        lr: float,
        target_names: list[str],
    ):
        super().__init__()
        self.core = core
        self.lr = lr
        self.target_names = list(target_names)
        self.T = len(self.target_names)

        self.val_corr: Dict[str, PearsonCorrCoef] = {name: PearsonCorrCoef() for name in self.target_names}
        self.test_corr: Dict[str, PearsonCorrCoef] = {name: PearsonCorrCoef() for name in self.target_names}

        self.save_hyperparameters(ignore=["core"])

    def forward(self, X: torch.Tensor, Ip: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        out = self.core(X, Ip, L)
        if out.dim() == 1:
            out = out.unsqueeze(1)
        return out

    def configure_optimizers(self):
        opt = RAdamScheduleFree(self.parameters(), lr=self.lr)
        opt.train()
        return opt

    def training_step(self, batch, _):
        X, Y, Ip, L = batch
        pred = self(X, Ip, L)

        losses = []
        for t, name in enumerate(self.target_names):
            loss_t = F.mse_loss(pred[:, t], Y[:, t])
            self.log(f"train/loss/{name}", loss_t, on_step=False, on_epoch=True, prog_bar=True)
            losses.append(loss_t)

        total_loss = torch.stack(losses).sum()
        return total_loss

    def on_validation_epoch_start(self):
        for m in self.val_corr.values():
            m.reset()

    def validation_step(self, batch, _):
        X, Y, Ip, L = batch
        pred = self(X, Ip, L)

        for t, name in enumerate(self.target_names):
            self.val_corr[name].update(pred[:, t], Y[:, t])

            loss_t = F.mse_loss(pred[:, t], Y[:, t])
            self.log(f"val/loss/{name}", loss_t, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        corr_sum = 0.0
        for name, metric in self.val_corr.items():
            r = metric.compute()
            self.log(f"val/pearsonr/{name}", r)
            corr_sum += float(r)

        self.log("val/accuracy", corr_sum)

    def on_test_epoch_start(self):
        for m in self.test_corr.values():
            m.reset()

    def test_step(self, batch, _):
        X, Y, Ip, L = batch
        pred = self(X, Ip, L)

        for t, name in enumerate(self.target_names):
            self.test_corr[name].update(pred[:, t], Y[:, t])

    def on_test_epoch_end(self):
        for name, metric in self.test_corr.items():
            r = metric.compute()
            self.log(f"test/pearsonr/{name}", r, prog_bar=True)
