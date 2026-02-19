from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int | None = None
    f_min: float = 0.0
    f_max: float = 8000.0
    n_mels: int = 80
    power: float = 1.0
    eps: float = 1e-5
    mel_scale: str = "slaney"
    norm: str | None = "slaney"


class MelSpectrogram(nn.Module):
    """
    HiFi-GAN style mel:
    - reflect-pad на (n_fft - hop)/2
    - center=False
    - возвращает log-mel
    """

    def __init__(self, config: MelSpectrogramConfig):
        super().__init__()
        self.config = config
        win_length = (
            config.win_length if config.win_length is not None else config.n_fft
        )

        # Для совпадения длин: pad = (n_fft - hop) / 2
        self.pad_size = (config.n_fft - config.hop_length) // 2
        if self.pad_size < 0:
            raise ValueError("n_fft должен быть >= hop_length")

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            n_fft=config.n_fft,
            win_length=win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=config.power,
            center=False,
            mel_scale=config.mel_scale,
            norm=config.norm,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: [T] | [1,T] | [B,T] | [B,1,T]
        return: [B, n_mels, T_mel] (log-mel)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1,T]
        if audio.dim() == 3:
            # [B,1,T] -> [B,T]
            audio = audio.squeeze(1)

        if audio.dim() != 2:
            raise ValueError(f"Expected audio dims 1/2/3, got {audio.shape}")

        # reflect-pad требует [B,1,T]
        if self.pad_size > 0:
            audio = F.pad(
                audio.unsqueeze(1), (self.pad_size, self.pad_size), mode="reflect"
            ).squeeze(1)

        mel = self.mel(audio)  # [B, n_mels, T]
        mel = mel.clamp_(min=self.config.eps).log_()
        return mel
