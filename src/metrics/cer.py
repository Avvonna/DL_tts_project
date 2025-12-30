from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    @torch.no_grad()
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        # [B, T]
        pred_ids = log_probs.detach().cpu().argmax(dim=-1)
        lengths = log_probs_length.detach().cpu().tolist()

        cers = []
        for ids, L, target_text in zip(pred_ids, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(ids[: int(L)])
            cers.append(calc_cer(target_text, pred_text))

        return float(sum(cers) / max(len(cers), 1))


class BeamCERMetric(BaseMetric):
    def __init__(
        self,
        text_encoder,
        beam_size=10,
        topk_per_timestep=20,
        beam_threshold=70.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = int(beam_size)
        self.topk_per_timestep = (
            int(topk_per_timestep) if topk_per_timestep is not None else None
        )
        self.beam_threshold = beam_threshold

    @torch.no_grad()
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        lengths = log_probs_length.detach().cpu().tolist()

        cers = []
        for i, (L, target_text) in enumerate(zip(lengths, text)):
            target_text = self.text_encoder.normalize_text(target_text)

            # [T, V] в log-домене
            lp = log_probs[i, : int(L)].detach().cpu()

            pred_text = self.text_encoder.ctc_beam_search(
                lp,
                beam_size=self.beam_size,
                topk_per_timestep=self.topk_per_timestep,
                beam_threshold=self.beam_threshold,
                input_type="log_probs",
            )
            cers.append(calc_cer(target_text, pred_text))

        return float(sum(cers) / max(len(cers), 1))
