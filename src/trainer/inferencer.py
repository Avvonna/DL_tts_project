from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker


class Inferencer:
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.

    Expected model output keys:
        - log_probs: Tensor [B, T, V] (log-softmax over vocab)
        - log_probs_length: Tensor [B] (valid lengths over T)

    Saves predictions as:
        save_path/<part>/<utt_id>.txt
    Optionally also writes:
        save_path/<part>/predictions.jsonl
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config,
        device: str,
        dataloaders: Dict[str, Any],
        text_encoder,
        save_path: Optional[Path] = None,
        metrics: Optional[Dict[str, list]] = None,
        batch_transforms: Optional[Dict[str, Dict[str, torch.nn.Module]]] = None,
        skip_model_load: bool = False,
    ):
        if not skip_model_load:
            ckpt = config.inferencer.get("from_pretrained")
            assert ckpt is not None, (
                "Provide checkpoint path in config.inferencer.from_pretrained "
                "or set skip_model_load=True"
            )

        self.config = config
        self.cfg_inf = config.inferencer

        self.device = device
        self.model = model.to(device)
        self.model.eval()

        self.text_encoder = text_encoder

        self.batch_transforms = batch_transforms or {"train": {}, "inference": {}}
        self.dataloaders = dict(dataloaders)

        self.save_path = Path(save_path) if save_path is not None else None

        self.metrics = metrics
        self.metric_tracker: Optional[MetricTracker] = None
        if self.metrics is not None and self.metrics.get("inference"):
            self.metric_tracker = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )

        if not skip_model_load:
            self._from_pretrained(self.cfg_inf.get("from_pretrained"))

        # decoding config
        self.decode_type = str(
            self.cfg_inf.get("decode_type", "greedy")
        )  # "greedy" | "beam"
        self.beam_size = int(self.cfg_inf.get("beam_size", 10))
        tpt = self.cfg_inf.get("topk_per_timestep", 20)
        self.topk_per_timestep = int(tpt) if tpt is not None else None
        self.beam_threshold = float(self.cfg_inf.get("beam_threshold", 70.0))
        self.save_both_decodes = bool(self.cfg_inf.get("save_both_decodes", False))

    def run_inference(self) -> Dict[str, Dict[str, float]]:
        part_logs: Dict[str, Dict[str, float]] = {}
        for part, dataloader in self.dataloaders.items():
            part_logs[part] = self._inference_part(part, dataloader)
        return part_logs

    @torch.inference_mode()
    def _inference_part(self, part: str, dataloader) -> Dict[str, float]:
        self.model.eval()

        if self.metric_tracker is not None:
            self.metric_tracker.reset()

        out_dir: Optional[Path] = None
        out_jsonl = None
        if self.save_path is not None:
            out_dir = self.save_path / part
            out_dir.mkdir(parents=True, exist_ok=True)
            out_jsonl = (out_dir / "predictions.jsonl").open("w", encoding="utf-8")

        sample_global_idx = 0

        for _, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
            batch = self._move_batch_to_device(batch)
            batch = self._transform_batch(batch, mode="inference")

            outputs = self.model(**batch)
            batch.update(outputs)

            if "log_probs" not in batch or "log_probs_length" not in batch:
                raise KeyError(
                    "Model output must contain 'log_probs' and 'log_probs_length'. "
                    f"Got keys: {list(batch.keys())}"
                )

            # 1. Декодирование
            pred_greedy = self._decode_greedy(
                batch["log_probs"], batch["log_probs_length"]
            )
            pred_beam = None

            if self.decode_type == "beam" or self.save_both_decodes:
                pred_beam = self._decode_beam(
                    batch["log_probs"], batch["log_probs_length"]
                )
                batch["pred_text_beam"] = pred_beam

            batch["pred_text_greedy"] = pred_greedy

            # Основной прогноз
            if self.decode_type == "beam" and pred_beam is not None:
                pred_texts = pred_beam
            else:
                pred_texts = pred_greedy

            pred_texts = batch["pred_text"]
            bsz = len(pred_texts)

            utt_ids = self._get_utt_ids(batch, default_start=sample_global_idx, bsz=bsz)

            # metrics: compute only if ground-truth texts exist
            if (
                self.metric_tracker is not None
                and self.metrics is not None
                and "text" in batch
            ):
                for met in self.metrics["inference"]:
                    self.metric_tracker.update(met.name, met(**batch))

            # save per-utterance txt + optional jsonl
            if out_dir is not None:
                for i in range(bsz):
                    pred_path = out_dir / f"{utt_ids[i]}.txt"
                    pred_path.write_text(pred_texts[i], encoding="utf-8")

                    if out_jsonl is not None:
                        rec = {"id": utt_ids[i], "pred_text": pred_texts[i]}
                        if "audio_path" in batch:
                            rec["audio_path"] = str(batch["audio_path"][i])
                        if "text" in batch:
                            rec["target_text"] = batch["text"][i]
                        if "pred_text_greedy" in batch:
                            rec["pred_text_greedy"] = batch["pred_text_greedy"][i]
                        if "pred_text_beam" in batch:
                            rec["pred_text_beam"] = batch["pred_text_beam"][i]

                        out_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

            sample_global_idx += bsz

        if out_jsonl is not None:
            out_jsonl.close()

        return {} if self.metric_tracker is None else self.metric_tracker.result()

    def _get_utt_ids(self, batch: dict, default_start: int, bsz: int) -> List[str]:
        # приоритет: explicit utt_id/id -> audio_path/path
        if "utt_id" in batch:
            v = batch["utt_id"]
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
            if torch.is_tensor(v):
                return [str(x.item()) for x in v]
            return [str(v)] * bsz

        if "id" in batch:
            v = batch["id"]
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
            if torch.is_tensor(v):
                return [str(x.item()) for x in v]
            return [str(v)] * bsz

        for key in ("audio_path", "path"):
            if key in batch:
                v = batch[key]
                if isinstance(v, (list, tuple)):
                    return [Path(p).stem for p in v]
                if torch.is_tensor(v):
                    return [str(x.item()) for x in v]
                if isinstance(v, str):
                    return [Path(v).stem] * bsz
                return [Path(v[i]).stem for i in range(bsz)]

        return [str(default_start + i) for i in range(bsz)]

    def _move_batch_to_device(self, batch: dict) -> dict:
        for tensor_name in self.cfg_inf.device_tensors:
            if tensor_name in batch and torch.is_tensor(batch[tensor_name]):
                batch[tensor_name] = batch[tensor_name].to(self.device)
        return batch

    def _transform_batch(self, batch: dict, mode: str) -> dict:
        transforms = self.batch_transforms.get(mode) or {}
        for tensor_name, transform in transforms.items():
            if tensor_name in batch and torch.is_tensor(batch[tensor_name]):
                batch[tensor_name] = transform(batch[tensor_name])
        return batch

    def _decode_greedy(
        self, log_probs: torch.Tensor, lengths: torch.Tensor
    ) -> List[str]:
        pred_ids = log_probs.argmax(dim=-1).detach().cpu()  # [B, T]
        lengths_cpu = lengths.detach().cpu().tolist()

        texts: List[str] = []
        for seq, L in zip(pred_ids, lengths_cpu):
            seq = seq[: int(L)].tolist()
            texts.append(self.text_encoder.ctc_decode(seq))
        return texts

    def _decode_beam(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        lp = log_probs.detach().cpu()  # [B, T, V] log-space
        lengths_cpu = lengths.detach().cpu().tolist()

        texts: List[str] = []
        for i, L in enumerate(lengths_cpu):
            texts.append(
                self.text_encoder.ctc_beam_search(
                    lp[i, : int(L)],  # [T, V]
                    beam_size=self.beam_size,
                    topk_per_timestep=self.topk_per_timestep,
                    beam_threshold=self.beam_threshold,
                    input_type="log_probs",
                )
            )
        return texts

    def _from_pretrained(self, pretrained_path):
        pretrained_path = str(pretrained_path)
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        state = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state)
