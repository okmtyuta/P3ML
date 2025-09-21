import torch


class BidirectionalSinusoidalPositionalEncoder(torch.nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int = 64, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, L, D = X.shape
        if L > self.pe.size(0):
            raise ValueError(f"sequence length {L} exceeds maximum supported length {self.pe.size(0)}")

        forward_pe = self.pe[:L, :D].to(X.device, X.dtype)
        indices = torch.arange(L - 1, -1, -1, device=self.pe.device)
        backward_pe = self.pe.index_select(0, indices)[:L, :D].to(X.device, X.dtype)

        forward_encoded = X + forward_pe
        backward_encoded = X + backward_pe

        return torch.cat([forward_encoded, backward_encoded], dim=2)
