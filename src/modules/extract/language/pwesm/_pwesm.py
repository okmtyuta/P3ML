from pathlib import Path
from typing import Literal, TypedDict

import h5py
import torch

from src.modules.extract.language._language import _Language
from src.modules.protein.protein import Protein


class _AminoAcidSource(TypedDict):
    name: str
    char: str
    representation: torch.Tensor


PWESMModelName = Literal["esm2", "esm1b"]


class _PWESMLanguage(_Language):
    def __init__(self, model_name: PWESMModelName):
        super().__init__()
        self._model_name: PWESMModelName = model_name
        self._source = self._load_source()

    def __call__(self, proteins: list[Protein]):
        for protein in proteins:
            protein.set_representations(self._convert(protein.seq))

        return proteins

    def _load_source(self) -> dict[str, _AminoAcidSource]:
        source: dict[str, _AminoAcidSource] = {}
        path = Path(__file__).parent / "pwesm.h5"
        with h5py.File(path, mode="r") as f:
            for key in f.keys():
                data = f[key]
                attrs = data.attrs

                char = attrs["char"]
                name = attrs["name"]

                representation = torch.from_numpy(data[self._model_name][:])
                source[key] = {"char": char, "name": name, "representation": representation}

        return source

    def _convert(self, chars: str):
        representations: list[torch.Tensor] = []
        for char in list(chars):
            representation = self._source[char]["representation"]
            representations.append(representation)

        return torch.stack(representations)
