import torch


class Concat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, Ip: torch.Tensor) -> torch.Tensor:
        ip = Ip.to(z.dtype)

        return torch.cat([z, ip], dim=1)
