import pytorch_lightning as plg
import torch
from pytorch_lightning.loggers import CSVLogger

from src.modules.dataloader.dataset import ProteinDataset, collate_fn
from src.modules.helper.helper import Helper
from src.modules.lit.lit import Lit
from src.modules.model.regressor import Regressor
from src.modules.protein.protein import Protein


def transfer_predict_hl_1(
    code: str, version: str, input_props: list[str], output_props: list[str], proteins: list[Protein]
):
    regressor = Regressor(input_dim=1280 + len(input_props), output_dim=len(output_props), hidden_dim=32, hidden_num=5)
    regressor.load_state_dict(torch.load(Helper.ROOT / "logs" / code / version / "weight.pt"))
    dataset = ProteinDataset(
        proteins=proteins,
        output_props=output_props,
        input_props=input_props,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )

    lit = Lit(model=regressor, lr=1e-3, output_props=output_props, input_props=input_props)

    logger = CSVLogger("logs", name=code)
    trainer = plg.Trainer(
        accelerator="cpu",
        devices=1,
        logger=logger,
        precision="32-true",
        log_every_n_steps=1,
        enable_checkpointing=False,
    )

    return trainer.test(lit, dataloaders=dataloader)
