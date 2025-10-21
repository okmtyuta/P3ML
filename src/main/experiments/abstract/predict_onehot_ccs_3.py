import json
import random
from pathlib import Path
from typing import Optional

import pytorch_lightning as plg
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.modules.dataloader.dataset import ProteinDataset, collate_fn
from src.modules.lit.lit import Lit
from src.modules.model.ccs_regressor import CCSRegressor_11
from src.modules.model.modules.concat import Concat
from src.modules.model.modules.head import FCNHead
from src.modules.model.modules.mean_aggregator import MeanAggregator
from src.modules.model.modules.sinusoidal_positional_encoder import SinusoidalPositionalEncoder
from src.modules.protein.protein import Protein


def predict_onehot_ccs_3(
    code: str,
    input_props: list[str],
    output_props: list[str],
    proteins: list[Protein],
    random_split_seed: Optional[int] = None,
):
    posenc = SinusoidalPositionalEncoder(d_model=1280, max_len=4096)
    aggregator = MeanAggregator()
    concat = Concat()
    head = FCNHead(input_dim=1280 + len(input_props), output_dim=len(output_props), hidden_dim=64, hidden_num=5)

    regressor = CCSRegressor_11(
        posenc=posenc,
        aggregator=aggregator,
        concat=concat,
        head=head,
    )

    dataset = ProteinDataset(
        proteins=proteins,
        output_props=output_props,
        input_props=input_props,
    )

    N = len(dataset)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    n_test = N - n_train - n_val

    if random_split_seed is None:
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
