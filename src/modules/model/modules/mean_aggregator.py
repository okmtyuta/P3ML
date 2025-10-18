import torch


class MeanAggregator(torch.nn.Module):
    def forward(self, X: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = (torch.arange(X.size(1), device=X.device)[None, :] < lengths[:, None]).float()
        return (X * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1)
