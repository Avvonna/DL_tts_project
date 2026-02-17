import torch


def resolve_device(cfg_device: str) -> str:
    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(cfg_device)
