import torch


class FCNHead(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_num: int):
        super().__init__()

        layers: list[torch.nn.Module] = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(hidden_num):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, output_dim))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)
