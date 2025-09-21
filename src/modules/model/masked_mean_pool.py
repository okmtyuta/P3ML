import torch


class MaskedMeanPool(torch.nn.Module):
    def forward(self, X: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, L, D = X.shape
        mask = torch.arange(L, device=X.device)[None, :] < lengths[:, None]  # (B,L)
        X = X * mask.unsqueeze(-1).to(X.dtype)
        denom = lengths.clamp_min(1).unsqueeze(1).to(X.dtype)
        return X.sum(dim=1) / denom
