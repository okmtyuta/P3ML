from typing import Literal, Optional, TypedDict

import torch

ProteinLanguageName = Literal["esm2", "esm1b", "pwesm1b", "pwesm2", "onehot"]
protein_language_names: list[ProteinLanguageName] = ["esm2", "esm1b", "pwesm1b", "pwesm2", "onehot"]


ProteinProps = dict[str, str | int | float]
ProteinPropName = str


class ProteinSource(TypedDict):
    key: str
    seq: str
    props: ProteinProps
    representations: Optional[torch.Tensor]
