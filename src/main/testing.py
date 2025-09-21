import pytorch_lightning as plg
import torch
from pytorch_lightning.loggers import CSVLogger

from src.modules.dataloader.dataset import ProteinDataset, collate_fn
from src.modules.lit.lit import LitMultiTask
from src.modules.model.ccs_regressor import CCSRegressor
from src.modules.protein.protein import Protein


def test(
    regressor: CCSRegressor,
    code: str,
    proteins: list[Protein],
    output_props: list[str],
    input_props: list[str],
    ckpt_path: str,
):
    plg.seed_everything(42)

    dataset = ProteinDataset(
        proteins=proteins,
        output_props=output_props,
        input_props=input_props,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    lit = LitMultiTask.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        core=regressor,
        output_props=output_props,
        input_props=input_props,
    )

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
