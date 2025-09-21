# src/models/embed.py
import torch


class OnehotEmbedded(torch.nn.Module):
    def __init__(self, aa_dim: int, out_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(aa_dim, out_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.proj(X)
