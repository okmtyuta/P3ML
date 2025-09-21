from pathlib import Path

from src.main.training import train
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
    head = Head(in_dim=65, out_dim=2)

    regressor = CCSRegressor(
        embed=embed,
        posenc=posenc,
        pool=pool,
        charge=charge,
        head=head,
    )

    proteins = ProteinList.from_hdf5("source/ishihama/onehot_normalized.h5").proteins
    code = Path(__file__).stem
    train(regressor=regressor, proteins=proteins, code=code, output_props=["rt"])


if __name__ == "__main__":
    main()
