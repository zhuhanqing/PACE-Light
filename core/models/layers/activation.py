from torch import nn

__all__ = ["SIREN"]


class SIREN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sin()
