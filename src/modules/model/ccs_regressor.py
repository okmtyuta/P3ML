import torch


class CCSRegressor_9(torch.nn.Module):
    def __init__(
        self,
        embed: torch.nn.Module,
        posenc: torch.nn.Module,
        aggregator: torch.nn.Module,
        concat: torch.nn.Module,
        head: torch.nn.Module,
    ):
        super().__init__()
        self.embed = embed
        self.posenc = posenc
        self.aggregator = aggregator
        self.concat = concat
        self.head = head

    def forward(
        self,
        X: torch.Tensor,
        Ip: torch.Tensor,
        L: torch.Tensor,
    ) -> torch.Tensor:
        y = self.embed(X)
        y = self.posenc(y)
        y = self.aggregator(y, L)
        y = self.concat(y, Ip)
        y = self.head(y)

        return y


class CCSRegressor_10(torch.nn.Module):
    def __init__(
        self,
        embed: torch.nn.Module,
        aggregator: torch.nn.Module,
        concat: torch.nn.Module,
        head: torch.nn.Module,
    ):
        super().__init__()
        self.embed = embed
        self.aggregator = aggregator
        self.concat = concat
        self.head = head

    def forward(
        self,
        X: torch.Tensor,
        Ip: torch.Tensor,
        L: torch.Tensor,
    ) -> torch.Tensor:
        y = self.embed(X)
        y = self.aggregator(y, L)
        y = self.concat(y, Ip)
        y = self.head(y)

        return y


class CCSRegressor_11(torch.nn.Module):
    def __init__(
        self,
        posenc: torch.nn.Module,
        aggregator: torch.nn.Module,
        concat: torch.nn.Module,
        head: torch.nn.Module,
    ):
        super().__init__()
        self.posenc = posenc
        self.aggregator = aggregator
        self.concat = concat
        self.head = head

    def forward(
        self,
        X: torch.Tensor,
        Ip: torch.Tensor,
        L: torch.Tensor,
    ) -> torch.Tensor:
        y = self.posenc(X)
        y = self.aggregator(y, L)
        y = self.concat(y, Ip)
        y = self.head(y)

        return y


class CCSRegressor_12(torch.nn.Module):
    def __init__(
        self,
        aggregator: torch.nn.Module,
        concat: torch.nn.Module,
        head: torch.nn.Module,
    ):
        super().__init__()
        self.aggregator = aggregator
        self.concat = concat
        self.head = head

    def forward(
        self,
        X: torch.Tensor,
        Ip: torch.Tensor,
        L: torch.Tensor,
    ) -> torch.Tensor:
        y = self.aggregator(X, L)
        y = self.concat(y, Ip)
        y = self.head(y)

        return y
