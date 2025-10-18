import torch


class Concat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, Ip: torch.Tensor) -> torch.Tensor:
        ip = Ip.to(x.dtype)

        return torch.cat([x, ip], dim=1)
