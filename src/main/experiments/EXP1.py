import json
import platform
import random
from pathlib import Path

import pytorch_lightning as plg
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.modules.dataloader.dataset import ProteinDataset, collate_fn
from src.modules.helper.helper import Helper
from src.modules.lit.lit import Lit
from src.modules.model.regressor import Regressor
from src.modules.protein.protein_list import ProteinList
from src.modules.slack_service import SlackService

code = Path(__file__).stem

output_props: list[str] = ["log_halflife"]
input_props: list[str] = []

regressor = Regressor(input_dim=1280, output_dim=1, hidden_dim=32, hidden_num=5)

proteins = ProteinList.from_hdf5(
    Helper.ROOT / "source" / "schwanhausser" / "esm2" / "logarithm.h5"
).proteins

dataset = ProteinDataset(
    proteins=proteins,
    output_props=output_props,
    input_props=input_props,
)


def train():
    # plg.seed_everything(42)

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
    torch.save(regressor.state_dict(), Path(logger.log_dir) / "weight.pt")

    results = trainer.test(lit, dataloaders=test_loader, ckpt_path=ckpt.best_model_path)
    result = results[0]

    result["random_split_seed"] = random_split_seed

    return result


def main() -> None:
    results: list[dict] = []

    server_name = platform.node()
    slack_service = SlackService()

    try:
        slack_service.send(f"[{server_name}] training started: {code}")
    except Exception as e:
        print(f"Slack notification was failed because of {e}")

    for _ in range(10):
        result = train()
        results.append(result)

        with open(Helper.ROOT / "logs" / code / "results.json", mode="w") as f:
            f.write(json.dumps(results))

    try:
        slack_service.send(f"[{server_name}] training end")
    except Exception as e:
        print(f"Slack notification was failed because of {e}: {code}")


if __name__ == "__main__":
    main()
