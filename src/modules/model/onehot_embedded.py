# src/models/embed.py
import torch


class OnehotEmbedded(torch.nn.Module):
    def __init__(self, aa_dim: int = 20, out_dim: int = 64, bias: bool = True):
        super().__init__()
        self.proj = torch.nn.Linear(aa_dim, out_dim, bias=bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.proj(X)
