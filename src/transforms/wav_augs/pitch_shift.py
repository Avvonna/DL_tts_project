import torch
import torchaudio
from torch import nn


class PitchShift(nn.Module):
    def __init__(self, p=0.5, sample_rate=16000, n_steps=4):
        super().__init__()
        self.p = p
        self.aug = torchaudio.transforms.PitchShift(sample_rate, n_steps)

    def forward(self, audio: torch.Tensor):
        if torch.rand(1) > self.p:
            return audio
        return self.aug(audio)
