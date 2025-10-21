import pytorch_lightning as plg
import torch
from pytorch_lightning.loggers import CSVLogger

from src.modules.dataloader.dataset import ProteinDataset, collate_fn
from src.modules.helper.helper import Helper
from src.modules.lit.lit import Lit
from src.modules.model.ccs_regressor import CCSRegressor_9
from src.modules.model.modules.concat import Concat
from src.modules.model.modules.head import FCNHead
from src.modules.model.modules.mean_aggregator import MeanAggregator
from src.modules.model.modules.onehot_embedded import OnehotEmbedded
from src.modules.model.modules.sinusoidal_positional_encoder import SinusoidalPositionalEncoder
from src.modules.protein.protein import Protein


def transfer_predict_onehot_ccs_1(
    code: str,
    from_code: str,
    from_version: str,
    input_props: list[str],
    output_props: list[str],
    proteins: list[Protein],
):
    embed = OnehotEmbedded(aa_dim=20, out_dim=64)
    posenc = SinusoidalPositionalEncoder(d_model=64, max_len=4096)
    aggregator = MeanAggregator()
    concat = Concat()
    head = FCNHead(input_dim=64 + len(input_props), output_dim=len(output_props), hidden_dim=64, hidden_num=5)

    regressor = CCSRegressor_9(
        embed=embed,
        posenc=posenc,
        aggregator=aggregator,
        concat=concat,
        head=head,
    )

    regressor.load_state_dict(torch.load(Helper.ROOT / "logs" / from_code / from_version / "weight.pt"))
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

    result = trainer.test(lit, dataloaders=dataloader)
    return result
