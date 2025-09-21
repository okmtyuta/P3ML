"""
language: pwesm1b
dataset: ishihama
input_props: ['charge']
output_props: ['ccs']
description: PWESM1bを利用した予測モデルをishihamaのデータで学習する。
"""

from pathlib import Path

import torch

from src.main.training import train
from src.modules.model.ccs_regressor import CCSRegressor
from src.modules.model.concat import Concat
from src.modules.model.head import Head
from src.modules.model.masked_mean_pool import MaskedMeanPool
from src.modules.model.sinusoidal_positional_encoder import SinusoidalPositionalEncoder
from src.modules.protein.protein_list import ProteinList


def EXP0002():
    embed = torch.nn.Identity()
    posenc = SinusoidalPositionalEncoder(d_model=1280, max_len=4096)
    pool = MaskedMeanPool()
    concat = Concat()
    head = Head(in_dim=1280 + 1, hidden_dim=64, out_dim=1)

    regressor = CCSRegressor(
        embed=embed,
        posenc=posenc,
        pool=pool,
        concat=concat,
        head=head,
    )

    proteins = ProteinList.from_hdf5("source/ishihama/pwesm1b.h5").proteins
    code = Path(__file__).stem
    train(
        regressor=regressor,
        proteins=proteins,
        code=code,
        output_props=["ccs"],
        input_props=["charge"],
    )
