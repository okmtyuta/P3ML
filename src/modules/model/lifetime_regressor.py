import torch

from src.modules.model.modules.concat import Concat
from src.modules.model.modules.mean_aggregator import MeanAggregator


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1280, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)


class LifetimeRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.head = Head()
        self.concat = Concat()
        self.aggregator = MeanAggregator()

    def forward(
        self,
        x: torch.Tensor,
        Ip: torch.Tensor,
        L: torch.Tensor,
    ) -> torch.Tensor:
        x = self.aggregator(x, L)
        x = self.concat(x, Ip)
        x = self.head(x)

        return x
