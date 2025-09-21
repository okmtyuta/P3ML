from pathlib import Path

import torch

from src.main.testing import test
from src.modules.model.ccs_regressor import CCSRegressor
from modules.model.concat import ChargeConcat
from src.modules.model.head import Head
from src.modules.model.masked_mean_pool import MaskedMeanPool
from src.modules.model.onehot_embedded import OnehotEmbedded
from src.modules.model.sinusoidal_positional_encoder import SinusoidalPositionalEncoder
from src.modules.protein.protein_list import ProteinList


def main():
    embed = OnehotEmbedded(aa_dim=20, out_dim=64)
    posenc = SinusoidalPositionalEncoder(d_model=64, max_len=4096)
    pool = MaskedMeanPool()
    charge = ChargeConcat()
    head = Head(in_dim=65, out_dim=1)

    regressor = CCSRegressor(
        embed=embed,
        posenc=posenc,
        pool=pool,
        charge=charge,
        head=head,
    )
    regressor.load_state_dict(torch.load("logs/EXP0001ccs-learn/version_0/ccs_core_state.pt"))

    proteins = ProteinList.from_hdf5("source/meier/onehot.h5").proteins

    code = Path(__file__).stem
    test(
        regressor=regressor,
        proteins=proteins,
        code=code,
        output_props=["ccs"],
        ckpt_path="logs/EXP0001ccs-learn/version_2/checkpoints/best.ckpt",
    )


if __name__ == "__main__":
    main()
