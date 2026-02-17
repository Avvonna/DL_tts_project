import torch


def match_audio_length(real_audio: torch.Tensor, fake_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Обрезка размерности аудио"""
    T = min(int(real_audio.size(-1)), int(fake_audio.size(-1)))
    return real_audio[..., :T], fake_audio[..., :T]

def assert_same_length(fake_audio: torch.Tensor, real_audio: torch.Tensor) -> None:
    """Проверка соответствия размерности"""
    if fake_audio.shape[-1] != real_audio.shape[-1]:
        raise RuntimeError(
            f"Length mismatch: fake={fake_audio.shape[-1]} real={real_audio.shape[-1]}"
        )
