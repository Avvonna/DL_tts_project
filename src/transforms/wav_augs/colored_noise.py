import torch
from torch import Tensor, nn


class ColoredNoise(nn.Module):
    """
    Adds colored noise to the audio signal.
    """

    def __init__(self, p=0.5, sample_rate=16000, snr_db=15):
        super().__init__()
        self.p = p
        self.snr_db = snr_db

    def forward(self, audio: Tensor):
        # audio: (Channels, Time)
        if torch.rand(1) > self.p:
            return audio

        # Генерируем шум
        noise = torch.randn_like(audio)

        # Считаем мощность сигнала и шума
        audio_power = audio.norm(p=2)
        noise_power = noise.norm(p=2)

        if noise_power == 0:
            return audio

        # Скейлим шум под нужный SNR
        snr = 10 ** (self.snr_db / 20)
        scale = (audio_power / noise_power) / snr

        return audio + noise * scale
