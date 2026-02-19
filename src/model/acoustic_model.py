from typing import Any

import torch
from transformers import AutoTokenizer, VitsModel


class HF_AcousticModel:
    """Обертка над HuggingFace VITS (MMS) для генерации аудио из текста."""

    def __init__(self, model_name: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = VitsModel.from_pretrained(model_name).to(device)  # type: ignore
        self.device = device
        self.sr = self.model.config.sampling_rate

    def generate_audio(self, text: str) -> tuple[Any, int]:
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model(**inputs).waveform  # (1, T)

        return output, self.sr
