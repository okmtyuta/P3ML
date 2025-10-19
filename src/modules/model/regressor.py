import torch

from src.modules.model.modules.concat import Concat
from src.modules.model.modules.head import FCNHead
from src.modules.model.modules.mean_aggregator import MeanAggregator


class Regressor(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_num: int):
        super().__init__()

        self.head = FCNHead(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, hidden_num=hidden_num)
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
