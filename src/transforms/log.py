import torch
from torch import nn


class Log(nn.Module):
    """
    Applies natural logarithm to the input tensor (usually spectrogram).
    x_log = log(x + eps)
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(x + self.eps)
