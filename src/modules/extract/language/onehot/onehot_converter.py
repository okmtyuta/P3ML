import torch

from src.modules.extract.language.source import amino_acid_source_mapping


class OnehotConverter:
    def __call__(self, seqs: list[str]) -> list[torch.Tensor]:
        representations: list[torch.Tensor] = []
        for seq in seqs:
            representation = torch.stack([amino_acid_source_mapping[aa]["onehot"] for aa in seq], dim=0)
            representations.append(representation)

        return representations
