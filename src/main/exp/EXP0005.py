from pathlib import Path

import torch

from src.main.training import train
from src.modules.model.ccs_regressor import CCSRegressor
from modules.model.concat import ChargeConcat
from src.modules.model.head import Head
from src.modules.model.masked_mean_pool import MaskedMeanPool
from src.modules.model.sinusoidal_positional_encoder import SinusoidalPositionalEncoder
from src.modules.protein.protein_list import ProteinList


def main():
    embed = torch.nn.Identity()
    posenc = SinusoidalPositionalEncoder(d_model=1280, max_len=4096)
    pool = MaskedMeanPool()
    charge = ChargeConcat()
    head = Head(in_dim=1281, out_dim=1)

    regressor = CCSRegressor(
        embed=embed,
        posenc=posenc,
        pool=pool,
        charge=charge,
        head=head,
    )

    proteins = ProteinList.from_hdf5("source/ishihama/esm1b.h5").proteins
    code = Path(__file__).stem
    train(regressor=regressor, proteins=proteins, code=code, output_props=["ccs"])


if __name__ == "__main__":
    main()
