import logging
import random
from typing import Any, Callable, Optional

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn import Module, ModuleDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.audio import assert_same_length, match_audio_length
from src.utils.timing import log_examples_per_sec


class GANTrainer(BaseTrainer):
    """
    Trainer for GAN-based TTS Vocoders
    """
    model: ModuleDict
    criterion: dict[str, Callable]
    optimizer: dict
    lr_scheduler: dict

    @property
    def _extra_log_keys(self) -> list[str]:
        return ["grad_norm_g", "grad_norm_d", "lr_g", "lr_d"]

    def __init__(
        self,
        generator: Module,
        discriminators: dict[str, Module],
        criterion: dict[str, Callable],
        metrics: dict[str, list[Callable]],
        config,
        device: str,
        dataloaders: dict[str, DataLoader],
        logger: logging.Logger,
        writer,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        mel_spec_transform: Optional[Module] = None,
        lr_scheduler_g: Optional[LRScheduler] = None,
        lr_scheduler_d: Optional[LRScheduler] = None,
        epoch_len: Optional[int] = None,
        skip_oom: bool = True,
        batch_transforms: Optional[dict[str, dict[str, Callable]]] = None,
    ):

        # Пакуем все в словари для BaseTrainer
        model_dict = {
            "generator": generator,
            **discriminators,
        }

        optimizer_dict = {
            "generator": optimizer_g,
            "discriminator": optimizer_d,
        }

        # Schedulers: собираем только те, что есть
        scheduler_dict = {}
        if lr_scheduler_g:
            scheduler_dict["generator"] = lr_scheduler_g
        if lr_scheduler_d:
            scheduler_dict["discriminator"] = lr_scheduler_d

        super().__init__(
            model=model_dict,
            criterion=criterion,
            metrics=metrics,
            config=config,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            optimizer=optimizer_dict,
            lr_scheduler=scheduler_dict if scheduler_dict else None,
            epoch_len=epoch_len,
            skip_oom=skip_oom,
            batch_transforms=batch_transforms,
        )

        # Создаем ссылки на объекты
        self.generator = self.model["generator"]
        # Все модели, кроме генератора - дискриминаторы (MPD, MSD и т.д.)
        self.discriminators = {k: v for k, v in self.model.items() if k != "generator"}
        self._d_params = [p for d in self.discriminators.values() for p in d.parameters()]

        self.optimizer_g = self.optimizer["generator"]
        self.optimizer_d = self.optimizer["discriminator"]

        # Достаем шедулеры обратно (если они были)
        if self.lr_scheduler is not None:
            self.lr_scheduler_g = self.lr_scheduler.get("generator")
            self.lr_scheduler_d = self.lr_scheduler.get("discriminator")
        else:
            self.lr_scheduler_g = None
            self.lr_scheduler_d = None

        # Преобразование в mel-спектрограмму нужно для валидации (Reconstruction Loss)
        self.mel_transform = mel_spec_transform.to(device) if mel_spec_transform else None

        # Настройки клиппинга градиентов из конфига
        self.grad_clip_g = self.config.trainer.get("grad_clip_g")
        self.grad_clip_d = self.config.trainer.get("grad_clip_d")

        self.use_amp = self.config.trainer.get("use_amp", False)
        if self.use_amp and self.device == 'cpu':
            self.logger.warning("AMP is enabled but device is CPU. This might be slow.")

        self.scaler = GradScaler(device=self.device, enabled=self.use_amp)

    def _set_requires_grad_params(self, params, flag: bool) -> None:
        """Функция установки requires_grad_"""
        for p in params:
            p.requires_grad_(flag)

    def _train_discriminator_step(self, batch: dict) -> tuple[torch.Tensor, dict, float]:
        """
        Один optimizer-step для дискриминаторов.
        - fake генерируется без графа (inference_mode)
        - loss_d.backward() строит градиенты только по D
        - grad_norm_d: либо clip_grad_norm_ (если задан max_norm), либо фактическая норма
        """
        self._set_requires_grad_params(self._d_params, True)
        self.optimizer_d.zero_grad(set_to_none=True)

        with autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
            with torch.no_grad():
                fake_detached = self.generator(batch["mel"])
            assert_same_length(fake_detached, batch["audio"])

            loss_d, loss_d_dict = self._compute_discriminator_loss(batch["audio"], fake_detached)

        self.scaler.scale(loss_d).backward()
        self.scaler.unscale_(self.optimizer_d)

        grad_norm_d = self._clip_grad_norm(model=self._d_params, max_norm=self.grad_clip_d)
        self.scaler.step(self.optimizer_d)
        self.scaler.update()

        return loss_d, loss_d_dict, float(grad_norm_d)

    def _train_generator_step(self, batch: dict) -> tuple[torch.Tensor, dict, float, torch.Tensor]:
        """
        Один optimizer-step для генератора.
        - параметры D заморожены (requires_grad=False), но forward D внутри loss'а
        всё равно нужен для adversarial + feature matching.
        - fake возвращаем наружу
        """
        self._set_requires_grad_params(self._d_params, False)
        self.optimizer_g.zero_grad(set_to_none=True)

        with autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
            fake = self.generator(batch["mel"])
            assert_same_length(fake, batch["audio"])

            loss_g, loss_g_dict = self._compute_generator_loss(batch["audio"], fake, batch["mel"])

        self.scaler.scale(loss_g).backward()
        self.scaler.unscale_(self.optimizer_g)

        grad_norm_g = self._clip_grad_norm(model=self.generator, max_norm=self.grad_clip_g)
        self.scaler.step(self.optimizer_g)
        self.scaler.update()

        return loss_g, loss_g_dict, float(grad_norm_g), fake

    def _update_train_metrics(
        self,
        loss_d: torch.Tensor,
        loss_g: torch.Tensor,
        loss_d_dict: dict,
        loss_g_dict: dict,
        grad_norm_d: float,
        grad_norm_g: float,
    ) -> None:
        """
        Обновляет трекер метрик.
        - loss_d / loss_g: скаляры per-batch
        - D/* и G/*: детализация компонент лосса (per sub-discriminator / per component)
        - grad_norm_*: значение после клиппинга (если клиппинг включен),
        иначе фактическая норма.
        """
        self.train_metrics.update("grad_norm_d", grad_norm_d)
        self.train_metrics.update("grad_norm_g", grad_norm_g)
        self.train_metrics.update("loss_d", float(loss_d.item()))
        self.train_metrics.update("loss_g", float(loss_g.item()))

        for name, val in loss_d_dict.items():
            v = float(val.item()) if torch.is_tensor(val) else float(val)
            self.train_metrics.update(f"D/{name}", v)

        for name, val in loss_g_dict.items():
            v = float(val.item()) if torch.is_tensor(val) else float(val)
            self.train_metrics.update(f"G/{name}", v)

    def _make_log_batch(self, batch: dict, fake_audio: torch.Tensor) -> dict:
        """Формирует батч для логирования."""
        out = dict(batch)
        out["audio_fake"] = fake_audio.detach().float().cpu()
        return out

    @log_examples_per_sec(
        get_n_examples=lambda self, epoch: self._n_train_examples(),
        get_mode=lambda self, epoch: "train",
    )
    def _train_epoch(self, epoch: int) -> dict[str, float]:
        # режим train и сброс метрик эпохи
        self.is_train = True
        self._set_model_mode(train=True)
        self.train_metrics.reset()

        pbar = tqdm(
            enumerate(self.train_dataloader),
            desc=f"Epoch {epoch} (Train)",
            total=self.epoch_len,
        )

        last_batch_gpu = None
        last_fake_gpu = None
        last_batch_idx = 0

        for batch_idx, batch in pbar:
            if batch_idx >= self.epoch_len:
                break

            try:
                # подготовка батча
                batch = self.move_batch_to_device(batch)
                batch = self.transform_batch(batch)

                # шаг D и G
                loss_d, loss_d_dict, grad_norm_d = self._train_discriminator_step(batch)
                loss_g, loss_g_dict, grad_norm_g, fake_audio = self._train_generator_step(batch)

                # обновление метрик
                self._update_train_metrics(
                    loss_d=loss_d,
                    loss_g=loss_g,
                    loss_d_dict=loss_d_dict,
                    loss_g_dict=loss_g_dict,
                    grad_norm_d=grad_norm_d,
                    grad_norm_g=grad_norm_g,
                )

                # сохраняем последний батч для финального логирования
                last_batch_gpu = batch
                last_fake_gpu = fake_audio.detach()
                last_batch_idx = batch_idx

                if batch_idx % self.log_step == 0:
                    log_batch = self._make_log_batch(batch, fake_audio)
                    self._log_step(self.train_metrics, batch_idx, epoch, batch=log_batch, mode="train")

                pbar.set_postfix({
                    "L_D": f"{loss_d.item():.3f}",
                    "L_G": f"{loss_g.item():.3f}",
                })

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self._handle_oom(batch_idx, e):
                    continue
                raise

        if self.lr_scheduler_g:
            self.lr_scheduler_g.step()
        if self.lr_scheduler_d:
            self.lr_scheduler_d.step()

        self.train_metrics.update("lr_g", float(self.optimizer_g.param_groups[0]["lr"]))
        self.train_metrics.update("lr_d", float(self.optimizer_d.param_groups[0]["lr"]))

        # Формируем last_batch для epoch_end логирования
        last_batch = None
        if last_batch_gpu is not None and last_fake_gpu is not None:
            last_batch = dict(last_batch_gpu)
            last_batch["audio_fake"] = last_fake_gpu.float()    # writer сам перенесет на cpu

        self._log_epoch_end(
            self.train_metrics,
            epoch,
            mode="train",
            last_batch=last_batch,
            last_batch_idx=last_batch_idx,
        )

        return self.train_metrics.result()


    def _compute_discriminator_loss(self, real_audio, fake_audio):
        """
        Forward всех дискриминаторов для real и fake.
        - fake_audio приходит без графа (detached)
        - градиенты считаются только по D
        """
        real_outs = {}
        fake_outs = {}

        for name, net in self.discriminators.items():
            real_outs[name] = net(real_audio)
            fake_outs[name] = net(fake_audio)

        loss_output = self.criterion["discriminator"](real_outs, fake_outs)
        if isinstance(loss_output, (tuple, list)):
            return loss_output[0], loss_output[1]
        return loss_output, {}

    def _compute_generator_loss(self, real_audio, fake_audio, mel):
        """
        Forward дискриминаторов для generator-loss.
        - real_outs считаем в no_grad(), потому что feature matching использует real fmaps
        как таргет; градиенты по real не нужны.
        - fake_outs считаем с графом по fake_audio (градиент идет в генератор),
        при этом параметры D заморожены requires_grad=False.
        """
        real_outs = {}
        fake_outs = {}

        for name, net in self.discriminators.items():
            with torch.no_grad():
                real_outs[name] = net(real_audio)
            fake_outs[name] = net(fake_audio)

        loss_output = self.criterion["generator"](
            audio=real_audio,
            audio_fake=fake_audio,
            disc_outputs_real=real_outs,
            disc_outputs_fake=fake_outs,
            mel=mel,
            mel_transform=self.mel_transform
        )

        if isinstance(loss_output, (tuple, list)):
            return loss_output[0], loss_output[1]
        return loss_output, {}

    def process_batch(self, batch: dict[str, Any], metrics: MetricTracker) -> dict[str, Any]:
        """
        Один batch в режиме evaluation/validation.
        - нет backward/step
        - считаем mel_loss в masked-виде по mel_length (игнорируем padding)
        - сохраняем mel_fake/mel_real/audio_fake для логов
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        self.generator.eval()
        for d in self.discriminators.values():
            d.eval()

        with torch.no_grad():

            # fake audio
            audio_fake = self.generator(batch["mel"])

            # выравниваем длины
            audio_real, audio_fake = match_audio_length(batch["audio"], audio_fake)

            # val loss D
            loss_d, loss_d_dict = self._compute_discriminator_loss(audio_real, audio_fake.detach())
            metrics.update("loss_d", float(loss_d))

            # val loss G
            loss_g, loss_g_dict = self._compute_generator_loss(audio_real, audio_fake, batch["mel"])
            metrics.update("loss_g", float(loss_g))

            # mel_loss
            if self.mel_transform:
                af = audio_fake.squeeze(1) if audio_fake.dim() == 3 else audio_fake  # (B, T)
                mel_fake = self.mel_transform(af) # log-mel

                mel_real = batch["mel"]          # (B, n_mels, T_max)
                mel_len = batch["mel_length"]    # (B,)

                mel_fake, mel_real = match_audio_length(mel_fake, mel_real)

                # маска по времени: (B, 1, T)
                t = torch.arange(mel_fake.size(-1), device=mel_real.device)[None, :]
                mask = (t < mel_len.to(mel_real.device)[:, None]).unsqueeze(1).float()

                diff = (mel_fake - mel_real).abs() * mask
                val_mel_loss = diff.sum() / (mask.sum() * mel_real.size(1) + 1e-8)
                metrics.update("mel_loss", float(val_mel_loss))

                # для логов
                batch["mel_fake"] = mel_fake.detach().float().cpu()
                batch["mel_real"] = mel_real.detach().float().cpu()

            # для логов
            batch["audio_fake"] = audio_fake.detach().float().cpu()

            for name, val in loss_d_dict.items():
                v = float(val) if torch.is_tensor(val) else float(val)
                metrics.update(f"D/{name}", v)
            for name, val in loss_g_dict.items():
                v = float(val) if torch.is_tensor(val) else float(val)
                metrics.update(f"G/{name}", v)

        return batch


    def _log_batch(self, batch_idx: int, batch: dict[str, Any], mode: str = "train"):
        """Логирование аудио и спектрограмм."""
        if self.writer is None:
            return

        sr = self.config.preprocess.get("audio", {}).get("sr", 22050)

        # По умолчанию берем 0ой элемент
        idx = 0
        use_random = self.config.writer.get("log_random_sample", False)

        if use_random:
            batch_size = 1
            if "audio" in batch:
                batch_size = batch["audio"].shape[0]
            elif "mel" in batch:
                batch_size = batch["mel"].shape[0]

            # Если батч больше 1, выбираем случайный индекс
            if batch_size > 1:
                idx = random.randint(0, batch_size - 1)

        # Логируем аудио
        if "audio" in batch:
            # На eval реальное аудио достаточно закинуть 1 раз - оно не меняется
            if mode == "train" or self._last_epoch <= 1:
                self.writer.add_audio("audio_real", batch["audio"][idx], sample_rate=sr)
        if "audio_fake" in batch:
            self.writer.add_audio("audio_fake", batch["audio_fake"][idx], sample_rate=sr)

        # Логируем спектрограммы
        def log_spec(name, tensor):
            # tensor: (n_mels, time), приводим к [0,1] для картинки
            spec = tensor.cpu()
            spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
            self.writer.add_image(f"{mode}/{name}", spec.unsqueeze(0))

        if "mel_fake" in batch:
            log_spec("spec_fake", batch["mel_fake"][idx])
        if "mel_real" in batch:
            log_spec("spec_real", batch["mel_real"][idx])
        elif "mel" in batch:
            log_spec("spec_real", batch["mel"][idx])
