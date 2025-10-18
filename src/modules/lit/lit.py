from typing import Dict

import polars as pl
import pytorch_lightning as plg
import torch
import torch.nn.functional as F
from schedulefree import RAdamScheduleFree
from torchmetrics import PearsonCorrCoef


class Lit(plg.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        output_props: list[str],
        input_props: list[str],
    ):
        super().__init__()
        self.model = model
        self.lr = lr

        self.output_props = output_props
        self.input_props = input_props

        self.train_corr: Dict[str, PearsonCorrCoef] = {name: PearsonCorrCoef() for name in self.output_props}
        self.val_corr: Dict[str, PearsonCorrCoef] = {name: PearsonCorrCoef() for name in self.output_props}
        self.test_corr: Dict[str, PearsonCorrCoef] = {name: PearsonCorrCoef() for name in self.output_props}

        self._test_records: list = []

        self.save_hyperparameters(ignore=["model"])

    def forward(self, X: torch.Tensor, Ip: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        out = self.model(X, Ip, L)
        if out.dim() == 1:
            out = out.unsqueeze(1)
        return out

    def configure_optimizers(self):
        opt = RAdamScheduleFree(self.parameters(), lr=self.lr)
        opt.train()
        return opt

    def on_train_epoch_start(self):
        for m in self.train_corr.values():
            m.reset()

    def training_step(self, batch, _):
        X, Y, Ip, L, _ = batch
        pred = self(X, Ip, L)

        losses = []
        for t, name in enumerate(self.output_props):
            self.train_corr[name].update(pred[:, t], Y[:, t])

            loss_t = F.mse_loss(pred[:, t], Y[:, t])
            self.log(
                f"train/loss/{name}",
                loss_t,
                on_step=False,
                on_epoch=True,
                batch_size=X.size(0),
            )
            losses.append(loss_t)

        total_loss = torch.stack(losses).sum()
        return total_loss

    def on_train_epoch_end(self):
        for name, metric in self.train_corr.items():
            r = metric.compute()
            self.log(f"train/pearsonr/{name}", r)

    def on_validation_epoch_start(self):
        for m in self.val_corr.values():
            m.reset()

    def validation_step(self, batch, _):
        X, Y, Ip, L, _ = batch
        pred = self(X, Ip, L)

        for t, name in enumerate(self.output_props):
            self.val_corr[name].update(pred[:, t], Y[:, t])

            loss_t = F.mse_loss(pred[:, t], Y[:, t])
            self.log(
                f"val/loss/{name}",
                loss_t,
                on_step=False,
                on_epoch=True,
                batch_size=X.size(0),
            )

    def on_validation_epoch_end(self):
        corr_sum = 0.0
        for name, metric in self.val_corr.items():
            r = metric.compute()
            self.log(f"val/pearsonr/{name}", r)
            corr_sum += float(r)

        self.log("val/accuracy", corr_sum, prog_bar=True)

    def on_test_epoch_start(self):
        for m in self.test_corr.values():
            m.reset()

    def on_test_start(self):
        self._test_records.clear()

    def test_step(self, batch, _):
        X, Y, Ip, L, proteins = batch
        pred = self(X, Ip, L)

        y_pred = pred.detach().cpu().numpy()

        for t, name in enumerate(self.output_props):
            self.test_corr[name].update(pred[:, t], Y[:, t])

            loss_t = F.mse_loss(pred[:, t], Y[:, t])
            self.log(
                f"test/loss/{name}",
                loss_t,
                on_step=False,
                on_epoch=True,
                batch_size=X.size(0),
            )

        for i, protein in enumerate(proteins):
            record = dict(protein._props)
            record["key"] = protein._key

            for j, name in enumerate(self.output_props):
                record[f"{name}_pred"] = float(y_pred[i, j])

            self._test_records.append(record)

    def on_test_epoch_end(self):
        for name, metric in self.test_corr.items():
            r = metric.compute()
            self.log(f"test/pearsonr/{name}", r, prog_bar=True)

        df = pl.DataFrame(self._test_records)
        df.write_csv(f"{self.logger.log_dir}/test_results.csv")
