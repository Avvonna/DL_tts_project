import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import torch
from numpy import inf
from omegaconf import DictConfig
from torch.nn import Module, ModuleDict
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from src.utils.timing import log_examples_per_sec


class BaseTrainer(ABC):
    """
    Base class for all trainers.
    Handles training loop, evaluation, checkpointing, and monitoring.

    Supports both single model/optimizer and dict-based (for GANs).
    """

    @property
    def _extra_log_keys(self) -> list[str]:
        """
        Возвращает список дополнительных ключей для логирования,
        которые не являются лоссами или метриками из конфига.
        По умолчанию только 'grad_norm'.
        """
        return ["grad_norm"]

    def __init__(
        self,
        model: Union[Module, dict[str, Module]],
        criterion: Union[Module, dict[str, Callable]],
        metrics: dict[str, list[Callable]],
        optimizer: Optional[Union[Optimizer, dict[str, Optimizer]]],
        lr_scheduler: Optional[Union[LRScheduler, dict[str, LRScheduler]]],
        config: DictConfig,
        device: str,
        dataloaders: dict[str, DataLoader],
        logger: logging.Logger,
        writer: Any,
        epoch_len: Optional[int] = None,
        skip_oom: bool = True,
        batch_transforms: Optional[dict[str, dict[str, Callable]]] = None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            text_encoder (CTCTextEncoder): text encoder.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.writer = writer

        # Validate config
        self._validate_config()
        self.log_step = self.cfg_trainer.get("log_step", 50)

        # Core components
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Normalize to ModuleDict if dict is passed
        if isinstance(model, dict):
            self.model = ModuleDict(model)

        if batch_transforms is None:
            batch_transforms = {"train": {}, "inference": {}}
        self.batch_transforms = batch_transforms

        # Dataloaders
        self.train_dataloader = dataloaders["train"]

        bs = (
            getattr(self.train_dataloader, "batch_size", None)
            or getattr(self.config.dataloader, "batch_size", None)
            or 1
        )
        self.train_batch_size = int(bs)

        if epoch_len is None:
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = int(epoch_len)

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # Epochs
        self._last_epoch = 0
        self.start_epoch = 1
        self.epochs = int(self.cfg_trainer.n_epochs)

        # Monitoring
        self.save_period = int(self.cfg_trainer.save_period)
        self.monitor = self.cfg_trainer.get("monitor", "off")

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0.0
            self.early_stop = inf
            self.mnt_metric = None
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]
            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # Metrics
        self.metrics = metrics
        self._setup_metric_trackers()

        # Checkpoint directory
        self.checkpoint_dir = (
            ROOT_PATH / self.cfg_trainer.save_dir / config.writer.run_name
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Resume / Pretrained
        if self.cfg_trainer.get("resume_from") is not None:
            self._resume_checkpoint(self.checkpoint_dir / self.cfg_trainer.resume_from)
        if self.cfg_trainer.get("from_pretrained") is not None:
            self._from_pretrained(self.cfg_trainer.get("from_pretrained"))

    def _setup_metric_trackers(self):
        """Хелпер для создания трекеров метрик"""

        def get_name(m):
            return getattr(m, "name", m.__name__)

        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *self._extra_log_keys,
            *[get_name(m) for m in self.metrics.get("train", [])],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[get_name(m) for m in self.metrics.get("inference", [])],
            writer=self.writer,
        )

    def _validate_config(self):
        """Validates that config has all required fields."""
        required_fields = {
            "trainer": ["n_epochs", "save_period", "save_dir"],
            "writer": ["loss_names", "run_name"],
        }

        for section, fields in required_fields.items():
            if not hasattr(self.config, section):
                raise ValueError(f"Config missing section: {section}")

            section_obj = getattr(self.config, section)
            for field in fields:
                if not hasattr(section_obj, field):
                    raise ValueError(
                        f"Config section '{section}' missing field: {field}"
                    )

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:
        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            logs: dict[str, float | int] = {"epoch": epoch}

            # Train epoch
            train_logs = self._train_epoch(epoch)
            logs.update(train_logs)

            # Evaluation epochs
            for part, dataloader in self.evaluation_dataloaders.items():
                eval_logs = self._evaluation_epoch(epoch, part, dataloader)
                logs.update(eval_logs)

            # Log all metrics
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # Monitoring
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            # Checkpointing
            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:
                break

    def _n_train_examples(self) -> int:
        """Returns total number of training examples per epoch."""
        return int(self.epoch_len) * int(self.train_batch_size)

    def _n_eval_examples(self, dataloader: DataLoader) -> int:
        """Returns total number of evaluation examples."""
        bs = int(getattr(dataloader, "batch_size", None) or 1)
        return int(len(dataloader)) * bs

    @abstractmethod
    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Этот метод должен реализовать наследник."""
        raise NotImplementedError()

    @log_examples_per_sec(
        get_n_examples=lambda self, epoch, part, dataloader: self._n_eval_examples(
            dataloader
        ),
        get_mode=lambda self, epoch, part, dataloader: str(part),
    )
    def _evaluation_epoch(
        self, epoch: int, part: str, dataloader: DataLoader
    ) -> dict[str, float]:
        self.is_train = False
        self._set_model_mode(train=False)

        # reset epoch+window
        self.evaluation_metrics.reset()

        last_batch = None
        last_batch_idx = 0
        total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader), desc=part, total=total_batches
            ):
                last_batch_idx = batch_idx
                last_batch = self.process_batch(batch, metrics=self.evaluation_metrics)

        # Логируем метрики за эпоху + последний батч
        self._log_epoch_end(
            self.evaluation_metrics,
            epoch,
            mode=part,
            last_batch=last_batch,
            last_batch_idx=last_batch_idx,
        )

        # Добавляем префикс к метрикам
        return {f"{part}_{k}": v for k, v in self.evaluation_metrics.result().items()}

    def _set_model_mode(self, train: bool):
        """
        Sets model(s) to train or eval mode.
        """
        if isinstance(self.model, ModuleDict):
            for model in self.model.values():
                model.train() if train else model.eval()
        elif isinstance(self.model, Module):
            self.model.train() if train else self.model.eval()
        else:
            raise TypeError(f"Unsupported model type: {type(self.model)}")

    def _monitor_performance(
        self, logs: dict[str, float | int], not_improved_count: int
    ) -> tuple[bool, bool, int]:
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False

        if self.mnt_mode == "off" or self.mnt_metric is None:
            return best, stop_process, not_improved_count

        current = logs.get(self.mnt_metric, None)
        if current is None:
            self.logger.warning(
                f"Metric '{self.mnt_metric}' not found. Monitoring disabled."
            )
            self.mnt_mode = "off"
            return best, stop_process, not_improved_count

        # check whether model performance improved or not
        improved = (
            (current <= self.mnt_best)
            if self.mnt_mode == "min"
            else (current >= self.mnt_best)
        )

        if improved:
            self.mnt_best = current
            not_improved_count = 0
            best = True
            self.logger.info(f"New best {self.mnt_metric}: {current:.4f}")
        else:
            not_improved_count += 1

        if not_improved_count >= self.early_stop:
            self.logger.info(
                f"Early stop: no improvement for {self.early_stop} epochs."
            )
            stop_process = True

        return best, stop_process, not_improved_count

    def _handle_oom(self, batch_idx: int, exception: Exception) -> bool:
        """
        Обработчик OOM.
        Если skip_oom=True, чистим кэш, обнуляем градиенты и идем дальше.
        """
        if not self.skip_oom:
            raise exception

        self.logger.warning(
            f"OOM on batch {batch_idx}. Skipping batch. " f"Error: {str(exception)}"
        )

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Zero all gradients
        if self.optimizer:
            if isinstance(self.optimizer, dict):
                for opt in self.optimizer.values():
                    opt.zero_grad(set_to_none=True)
            else:
                self.optimizer.zero_grad(set_to_none=True)
        return True

    def move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_name in self.cfg_trainer.device_tensors:
            if tensor_name in batch and torch.is_tensor(batch[tensor_name]):
                batch[tensor_name] = batch[tensor_name].to(self.device)
        return batch

    def transform_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = (self.batch_transforms or {}).get(transform_type) or {}
        for tensor_name, transform in transforms.items():
            if tensor_name in batch and torch.is_tensor(batch[tensor_name]):
                batch[tensor_name] = transform(batch[tensor_name])
        return batch

    def _clip_grad_norm(
        self,
        model: Union[Module, Iterable[torch.Tensor], None] = None,
        max_norm: Optional[float] = None,
    ):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        # Если модель не передана, берем "родную" модель трейнера
        if model is None:
            target = self.model
        else:
            target = model

        if isinstance(target, ModuleDict):
            # Если это словарь, собираем параметры со всех
            params = [p for m in target.values() for p in m.parameters()]
        elif isinstance(target, Module):
            # Если обычная модель, берем её параметры
            params = list(target.parameters())
        else:
            # Список параметров или генератор
            params = [p for p in target if isinstance(p, torch.Tensor)]

        # Если max_norm нет, ищем в конфиге
        if max_norm is None:
            max_norm = self.config.trainer.get("max_grad_norm")

        if max_norm:
            return clip_grad_norm_(params, max_norm).item()

        # Если клипать не надо, считаем норму
        return self._get_grad_norm_from_params(params)

    @torch.no_grad()
    def _get_grad_norm_from_params(self, parameters, norm_type: int = 2) -> float:
        parameters = list(parameters)

        # Оставляем только те параметры, у которых есть градиент (для соответсвия типов)
        grads = [p.grad.detach() for p in parameters if p.grad is not None]

        if len(grads) == 0:
            return 0.0

        # Считаем общую норму вектора
        total_norm = torch.norm(
            torch.stack([torch.norm(g, norm_type) for g in grads]),
            norm_type,
        )
        return total_norm.item()

    def _log_step(
        self,
        metric_tracker: MetricTracker,
        batch_idx: int,
        epoch: int,
        batch: Optional[dict[str, Any]] = None,
        mode: str = "train",
    ):
        """
        Универсальный метод для логирования шага.
        Пишет скаляры, медиа и сбрасывает window-метрики.
        """
        if self.writer is None:
            return

        self.writer.set_step((epoch - 1) * (self.epoch_len + 1) + batch_idx, mode)

        # Логируем числа
        self._log_scalars_window(metric_tracker)

        # Логируем медиа
        if batch is not None:
            self._log_batch(batch_idx, batch, mode)

        # Сбрасываем накопление
        metric_tracker.reset_window()

    def _log_epoch_end(
        self,
        metric_tracker: MetricTracker,
        epoch: int,
        mode: str = "train",
        last_batch: Optional[dict[str, Any]] = None,
        last_batch_idx: int = 0,
    ):
        """
        Логирование в конце эпохи: финальные метрики + последний батч.
        """
        if self.writer is None:
            return

        self.writer.set_step((epoch - 1) * (self.epoch_len + 1) + self.epoch_len, mode)

        if mode == "train":
            self.writer.add_scalar("trainer/progress", float(epoch))

        # Логируем агрегированные метрики за эпоху
        self._log_scalars_epoch(metric_tracker)

        # Логируем последний батч
        if last_batch is not None:
            self._log_batch(last_batch_idx, last_batch, mode=mode)

    def _log_scalars_window(self, metric_tracker: MetricTracker):
        """Логирует оконные средние (для шага)."""
        if self.writer is None:
            return

        scalars = {}
        for metric_name in metric_tracker.keys():
            if metric_tracker.count_window(metric_name) > 0:
                scalars[metric_name] = float(metric_tracker.avg_window(metric_name))

        if scalars:
            self.writer.add_scalars(scalars)

    def _log_scalars_epoch(self, metric_tracker: MetricTracker):
        """Логирует средние за эпоху."""
        if self.writer is None:
            return

        scalars = {}
        for metric_name in metric_tracker.keys():
            if metric_tracker.count(metric_name) > 0:
                scalars[metric_name] = float(metric_tracker.avg(metric_name))

        if scalars:
            self.writer.add_scalars(scalars)

    @abstractmethod
    def process_batch(
        self, batch: dict[str, Any], metrics: MetricTracker
    ) -> dict[str, Any]:
        """Расчет лосса и апдейт метрик. Реализуется в конкретном тренере."""
        raise NotImplementedError()

    @abstractmethod
    def _log_batch(self, batch_idx: int, batch: dict[str, Any], mode: str = "train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.
        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        raise NotImplementedError()

    def _get_checkpoint_data(self) -> dict[str, Any]:
        """
        Gets checkpoint data to save. Override in child class for additional state.

        Returns:
            dict with checkpoint data.
        """
        checkpoint = {
            "monitor_best": self.mnt_best,
            "config": self.config,
        }

        # Сохранение state_dict
        if isinstance(self.model, ModuleDict):
            checkpoint["state_dict"] = {
                k: v.state_dict() for k, v in self.model.items()
            }
        elif isinstance(self.model, Module):
            checkpoint["state_dict"] = self.model.state_dict()
        else:
            raise TypeError(f"Unsupported model type: {type(self.model)}")

        # optimizer (single or dict)
        if isinstance(self.optimizer, dict):
            checkpoint["optimizer"] = {
                k: v.state_dict() for k, v in self.optimizer.items()
            }
        elif self.optimizer is not None:
            checkpoint["optimizer"] = self.optimizer.state_dict()

        # scheduler (single or dict)
        if isinstance(self.lr_scheduler, dict):
            checkpoint["lr_scheduler"] = {
                k: v.state_dict() for k, v in self.lr_scheduler.items()
            }
        elif self.lr_scheduler is not None:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()

        return checkpoint

    def _load_checkpoint_data(self, checkpoint: dict[str, Any]):
        """
        Loads checkpoint data. Override in child class for additional state.

        Args:
            checkpoint: Loaded checkpoint dict.
        """
        # Load model
        if "state_dict" in checkpoint and checkpoint["state_dict"] is not None:
            if isinstance(self.model, ModuleDict):
                # dict of models
                for name, state_dict in checkpoint["state_dict"].items():
                    if name in self.model:
                        self.model[name].load_state_dict(state_dict)
            elif isinstance(self.model, Module):
                # Single model
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                raise TypeError(f"Unsupported model type: {type(self.model)}")

        # Load optimizer
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            if isinstance(self.optimizer, dict):
                for name, state_dict in checkpoint["optimizer"].items():
                    if name in self.optimizer:
                        self.optimizer[name].load_state_dict(state_dict)
            elif self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load scheduler
        if "lr_scheduler" in checkpoint and checkpoint["lr_scheduler"] is not None:
            if isinstance(self.lr_scheduler, dict):
                for name, state_dict in checkpoint["lr_scheduler"].items():
                    if name in self.lr_scheduler:
                        self.lr_scheduler[name].load_state_dict(state_dict)
            elif self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def _save_checkpoint(
        self, epoch: int, save_best: bool = False, only_best: bool = False
    ):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """

        state = self._get_checkpoint_data()
        state["epoch"] = epoch

        # Save regular checkpoint
        if not (only_best and save_best):
            filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
            torch.save(state, filename)
            if (
                getattr(self.config.writer, "log_checkpoints", False)
                and self.writer is not None
            ):
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")

        # Save best checkpoint
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if (
                getattr(self.config.writer, "log_checkpoints", False)
                and self.writer is not None
            ):
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path: Path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """

        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(
            resume_path, map_location=self.device, weights_only=False
        )
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.mnt_best = checkpoint.get("monitor_best", self.mnt_best)
        self._load_checkpoint_data(checkpoint)
        self.logger.info(f"Checkpoint loaded. Resume from epoch {self.start_epoch}")

    def _from_pretrained(self, pretrained_path: str):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        self.logger.info(f"Loading pretrained weights: {pretrained_path}")
        checkpoint = torch.load(
            pretrained_path, map_location=self.device, weights_only=False
        )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(
                "Pretrained checkpoint must be a dict or contain 'state_dict'"
            )

        if isinstance(self.model, ModuleDict):
            for k, v in state_dict.items():
                if k in self.model:
                    self.model[k].load_state_dict(v)
        elif isinstance(self.model, Module):
            self.model.load_state_dict(state_dict)
        else:
            raise TypeError(f"Unsupported model type: {type(self.model)}")
