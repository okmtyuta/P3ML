from typing import List

import torch

from src.modules.protein.protein import Protein


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        proteins: List[Protein],
        input_props: list[str],
        output_props: list[str],
    ):
        self._proteins = proteins
        self._input_props = tuple(input_props)
        self._output_props = tuple(output_props)

    def __len__(self) -> int:
        return len(self._proteins)

    def __getitem__(self, i: int):
        protein = self._proteins[i]

        x = torch.as_tensor(protein.representations, dtype=torch.float32)
        y = torch.tensor([protein.read_props(t) for t in self._output_props], dtype=torch.float32)

        ip = torch.tensor([protein.read_props(t) for t in self._input_props], dtype=torch.float32)

        return x, y, ip, protein


def collate_fn(batch):
    xs, ys, ips, proteins = zip(*batch)

    L = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    Lmax, A = int(L.max()), xs[0].shape[1]

    X = torch.zeros(len(xs), Lmax, A, dtype=torch.float32)
    for i, x in enumerate(xs):
        X[i, : x.shape[0]] = x.to(torch.float32)

    Y = torch.stack([torch.as_tensor(y, dtype=torch.float32) for y in ys])
    Ip = torch.stack([torch.as_tensor(ip, dtype=torch.float32) for ip in ips])

    return X, Y, Ip, L, proteins
