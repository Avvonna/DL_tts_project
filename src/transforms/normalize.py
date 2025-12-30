from torch import nn


class Normalize1D(nn.Module):
    """
    Normalizes the input tensor (spectrogram) to have zero mean and unit variance.
    Usually applied per-sample over the time dimension.
    """

    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        # x shape: (Batch, Freq, Time) или (Freq, Time)
        # Мы нормализуем по временной размерности (последней)

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6  # 1e-6 для стабильности

        return (x - mean) / std * self.std + self.mean
