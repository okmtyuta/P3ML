from pathlib import Path

import pytorch_lightning as plg
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from modules.model.regressor import LifetimeRegressor
from src.modules.dataloader.dataset import ProteinDataset, collate_fn
from src.modules.lit.lit import Lit
from src.modules.protein.protein_list import ProteinList


def main():
    regressor = LifetimeRegressor()

    proteins = ProteinList.from_hdf5("source/schwanhausser/saprot/saprot_logarithm.h5").proteins

    code = Path(__file__).stem

    plg.seed_everything(42)

    output_props = ["log_halflife"]
    input_props = []

    dataset = ProteinDataset(
        proteins=proteins,
        output_props=output_props,
        input_props=input_props,
    )

    N = len(dataset)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    n_test = N - n_train - n_val

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    lit = Lit(model=regressor, lr=1e-3, output_props=output_props, input_props=input_props)

    early_stop = EarlyStopping(monitor="val/accuracy", mode="max", patience=100)
    ckpt = ModelCheckpoint(monitor="val/accuracy", mode="max", save_top_k=1, filename="best")

    logger = CSVLogger("logs", name=code)

    trainer = plg.Trainer(
        accelerator="cpu",
        devices=1,
        callbacks=[early_stop, ckpt],
        logger=logger,
        precision="32-true",
        log_every_n_steps=1,
    )

    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit, dataloaders=test_loader, ckpt_path=ckpt.best_model_path)

    torch.save(regressor.state_dict(), f"{logger.log_dir}/weight.pt")


if __name__ == "__main__":
    main()
