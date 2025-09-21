import random
import secrets
from pathlib import Path
from typing import Optional, Self

import h5py
import polars as pl
from tqdm import tqdm

from src.modules.protein.protein import Protein
from src.modules.protein.types import ProteinProps


class ProteinList:
    def __init__(self, proteins: list[Protein]) -> None:
        self._proteins = proteins

    @property
    def proteins(self):
        return self._proteins

    @classmethod
    def from_csv(cls, path: Path) -> "ProteinList":
        df = pl.read_csv(path)

        proteins = []
        for row in df.iter_rows(named=True):
            key = str(row["index"])

            props: ProteinProps = {k: v for k, v in row.items() if k != "index"}

            protein = Protein(key=key, props=props, representations=None)
            proteins.append(protein)

        return ProteinList(proteins=proteins)

    def shuffle(self, seed: Optional[int] = None) -> Self:
        if seed is None:
            seed = secrets.randbits(64)

        rng = random.Random(seed)
        rng.shuffle(self._proteins)

        return self

    def to_hdf5_group(self, group: h5py.Group):
        for protein in self._proteins:
            protein_group = group.create_group(name=str(protein._key))
            protein.to_hdf5_group(group=protein_group)

    def to_hdf5(self, path: Path):
        with h5py.File(name=path, mode="w") as f:
            self.to_hdf5_group(f)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group):
        proteins: list[Protein] = []
        for protein_group in tqdm(group.values()):
            protein = Protein.from_hdf5_group(group=protein_group)
            proteins.append(protein)

        return ProteinList(proteins=proteins)

    @classmethod
    def from_hdf5(cls, path: Path):
        with h5py.File(name=path, mode="r") as f:
            protein_list = cls.from_hdf5_group(group=f)
            return protein_list
