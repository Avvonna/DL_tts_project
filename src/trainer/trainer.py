import logging
from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()

            grad_norm = self._clip_grad_norm()
            batch["grad_norm"] = grad_norm
            metrics.update("grad_norm", grad_norm)

            self.optimizer.step()
            self._scheduler_step_batch()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.
        """
        # Логируем спектрограммы в обоих режимах
        self.log_spectrogram(**batch)

        # Предсказания логируем только на inference
        if mode != "train":
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self,
        text=None,
        log_probs=None,
        log_probs_length=None,
        audio_path=None,
        examples_to_log=10,
        **batch,
    ):
        if log_probs is None or log_probs_length is None:
            return

        decoding_cfg = self.config.get("decoding", {})
        beam_size = decoding_cfg.get("beam_size", 1)
        topk = decoding_cfg.get("topk_per_timestep", None)
        threshold = decoding_cfg.get("beam_threshold", 70.0)

        log_probs_length_cpu = log_probs_length.detach().cpu().tolist()
        log_probs_cpu = log_probs.detach().cpu()

        limit = min(len(log_probs_cpu), examples_to_log)

        # Argmax (всегда)
        argmax_inds = log_probs_cpu.argmax(-1).numpy()
        argmax_inds = [
            inds[: int(L)] for inds, L in zip(argmax_inds, log_probs_length_cpu)
        ]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

        # Beam Search
        beam_texts = None
        if beam_size is not None and beam_size > 1:
            beam_texts = []
            for i in range(limit):
                L = int(log_probs_length_cpu[i])
                lp = log_probs_cpu[i, :L]
                beam_texts.append(
                    self.text_encoder.ctc_beam_search(
                        lp,
                        beam_size=beam_size,
                        topk_per_timestep=topk,
                        beam_threshold=threshold,
                        input_type="log_probs",
                    )
                )

        if audio_path is None:
            audio_path = [f"sample_{i}" for i in range(len(argmax_texts))]

        if text is None:
            text = [""] * len(argmax_texts)

        rows = {}
        for i in range(limit):
            target = self.text_encoder.normalize_text(text[i])
            pred_argmax = argmax_texts[i]

            wer_argmax = calc_wer(target, pred_argmax) * 100
            cer_argmax = calc_cer(target, pred_argmax) * 100

            row_dict = {
                "target": target,
                "argmax_pred": pred_argmax,
                "wer_argmax": wer_argmax,
                "cer_argmax": cer_argmax,
            }

            if beam_texts is not None:
                pred_beam = beam_texts[i]
                row_dict["beam_pred"] = pred_beam
                row_dict["wer_beam"] = calc_wer(target, pred_beam) * 100
                row_dict["cer_beam"] = calc_cer(target, pred_beam) * 100

            rows[Path(audio_path[i]).name] = row_dict

        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
