import torch


class CCSRegressor(torch.nn.Module):
    def __init__(
        self,
        embed: torch.nn.Module,
        posenc: torch.nn.Module,
        pool: torch.nn.Module,
        charge: torch.nn.Module,
        head: torch.nn.Module,
    ):
        super().__init__()
        self.embed = embed
        self.posenc = posenc
        self.pool = pool
        self.charge = charge
        self.head = head

    def forward(
        self,
        X: torch.Tensor,
        Ip: torch.Tensor,
        L: torch.Tensor,
    ) -> torch.Tensor:
        y = self.embed(X)
        y = self.posenc(y)
        y = self.pool(y, L)
        y = self.charge(y, Ip)
        y = self.head(y)

        return y
