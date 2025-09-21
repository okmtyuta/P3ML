from typing import TypedDict

import torch


class AminoAcidSource(TypedDict):
    name: str
    char: str
    onehot: torch.Tensor


amino_acid_source_mapping: dict[str, AminoAcidSource] = {
    "A": {
        "name": "Alanine",
        "char": "A",
        "onehot": torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "R": {
        "name": "Arginine",
        "char": "R",
        "onehot": torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "N": {
        "name": "Asparagine",
        "char": "N",
        "onehot": torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "D": {
        "name": "Aspartate",
        "char": "D",
        "onehot": torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "C": {
        "name": "Cysteine",
        "char": "C",
        "onehot": torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "Q": {
        "name": "Glutamine",
        "char": "Q",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "E": {
        "name": "Glutamate",
        "char": "E",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "G": {
        "name": "Glycine",
        "char": "G",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "H": {
        "name": "Histidine",
        "char": "H",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "I": {
        "name": "Isoleucine",
        "char": "I",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "L": {
        "name": "Leucine",
        "char": "L",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "K": {
        "name": "Lysine",
        "char": "K",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "M": {
        "name": "Methionine",
        "char": "M",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "F": {
        "name": "Phenylalanine",
        "char": "F",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "P": {
        "name": "Proline",
        "char": "P",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float32),
    },
    "S": {
        "name": "Serine",
        "char": "S",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32),
    },
    "T": {
        "name": "Threonine",
        "char": "T",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32),
    },
    "W": {
        "name": "Tryptophan",
        "char": "W",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float32),
    },
    "Y": {
        "name": "Tyrosine",
        "char": "Y",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float32),
    },
    "V": {
        "name": "Valine",
        "char": "V",
        "onehot": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32),
    },
}
