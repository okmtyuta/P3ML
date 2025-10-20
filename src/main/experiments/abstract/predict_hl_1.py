import json
import platform
import random
from pathlib import Path

import pytorch_lightning as plg
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.modules.dataloader.dataset import ProteinDataset, collate_fn
from src.modules.lit.lit import Lit
from src.modules.model.regressor import Regressor
from src.modules.protein.protein import Protein
from src.modules.slack_service import SlackService


def predict_hl_1(code: str, input_props: list[str], output_props: list[str], proteins: list[Protein]):
    server_name = platform.node()
    slack_service = SlackService()

    try:
        slack_service.send(f"[{server_name}] training started: {code}")
    except Exception as e:
        print(f"Slack notification was failed because of {e}")

    regressor = Regressor(input_dim=1280 + len(input_props), output_dim=len(output_props), hidden_dim=32, hidden_num=5)
    dataset = ProteinDataset(
        proteins=proteins,
        output_props=output_props,
        input_props=input_props,
    )

    N = len(dataset)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    n_test = N - n_train - n_val

    random_split_seed = random.randint(0, 2**32 - 1)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(random_split_seed),
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
    torch.save(regressor.state_dict(), Path(logger.log_dir) / "weight.pt")

    meta = {"random_split_seed": random_split_seed}
    with open(Path(logger.log_dir) / "meta.json", mode="w") as f:
        f.write(json.dumps(meta))

    try:
        slack_service.send(f"[{server_name}] training end: {code}")
    except Exception as e:
        print(f"Slack notification was failed because of {e}: {code}")
