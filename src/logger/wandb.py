from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import torch


class WandBWriter:
    """
    Class for experiment tracking via WandB.

    See https://docs.wandb.ai/.
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        entity=None,
        run_id=None,
        run_name=None,
        mode: Literal["online", "offline", "disabled", "shared"] | None = "online",
        **kwargs,
    ):
        """
        API key is expected to be provided by the user in the terminal.

        Args:
            logger (Logger): logger that logs output.
            project_config (dict): config for the current experiment.
            project_name (str): name of the project inside experiment tracker.
            entity (str | None): name of the entity inside experiment
                tracker. Used if you work in a team.
            run_id (str | None): the id of the current run.
            run_name (str | None): the name of the run. If None, random name
                is given.
            mode (str): if online, log data to the remote server. If
                offline, log locally.
        """
        try:
            import wandb

            wandb.login()

            self.run_id = run_id

            resume_epoch = kwargs.get("resume_epoch")
            epoch_len = kwargs.get("epoch_len")
            enable_rewind = kwargs.get("rewind", False)
            resume_arg = "allow"
            resume_from_arg = None

            if (
                enable_rewind
                and resume_epoch is not None
                and run_id is not None
                and epoch_len is not None
            ):
                rewind_step = (resume_epoch - 1) * (epoch_len + 1)
                logger.info(
                    f"Rewinding wandb run {run_id} to step {rewind_step} "
                    f"(epoch {resume_epoch}, epoch_len {epoch_len})"
                )
                resume_arg = None
                resume_from_arg = f"{run_id}?_step={rewind_step}"

            wandb.init(
                project=project_name,
                entity=entity,
                config=project_config,
                name=run_name,
                resume=resume_arg,
                id=run_id if resume_from_arg is None else None,
                resume_from=resume_from_arg,
                mode=mode,
                save_code=kwargs.get("save_code", False),
            )
            self.wandb = wandb

        except ImportError:
            logger.warning("For use wandb install it via \n\t pip install wandb")

        self.step = 0
        # the mode is usually equal to the current partition name
        # used to separate Partition1 and Partition2 metrics
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Define current step and mode for the tracker.

        Calculates the difference between method calls to monitor
        training/evaluation speed.

        Args:
            step (int): current step.
            mode (str): current mode (partition name).
        """
        self.mode = mode
        self.step = int(step)

    def _object_name(self, object_name):
        """
        Update object_name (scalar, image, etc.) with the
        current mode (partition name). Used to separate metrics
        from different partitions.

        Args:
            object_name (str): current object name.
        Returns:
            object_name (str): updated object name.
        """
        return f"{object_name}_{self.mode}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        """
        Log checkpoints to the experiment tracker.

        The checkpoints will be available in the files section
        inside the run_name dir.

        Args:
            checkpoint_path (str): path to the checkpoint file.
            save_dir (str): path to the dir, where checkpoint is saved.
        """
        self.wandb.save(checkpoint_path, base_path=save_dir)

    def add_scalar(self, scalar_name, scalar):
        """
        Log a scalar to the experiment tracker.

        Args:
            scalar_name (str): name of the scalar to use in the tracker.
            scalar (float): value of the scalar.
        """
        self.wandb.log(
            {
                self._object_name(scalar_name): scalar,
            },
            step=self.step,
        )

    def add_scalars(self, scalars):
        """
        Log several scalars to the experiment tracker.

        Args:
            scalars (dict): dict, containing scalar name and value.
        """
        self.wandb.log(
            {
                self._object_name(scalar_name): scalar
                for scalar_name, scalar in scalars.items()
            },
            step=self.step,
        )

    def add_image(self, image_name, image):
        """
        Log an image to the experiment tracker.

        Args:
            image_name (str): name of the image to use in the tracker.
            image (Path | ndarray | Image | Tensor): image data.
        """
        if hasattr(image, "detach"):
            image = image.detach().cpu()

            # Исправление размерности
            if image.dim() == 2:
                image = image.unsqueeze(0)

            # Нормализация
            if image.dtype in (float, torch.float16, torch.float32, torch.float64):
                min_val = image.min()
                max_val = image.max()
                if max_val > min_val:  # защита от деления на 0
                    image = (image - min_val) / (max_val - min_val)

        self.wandb.log(
            {self._object_name(image_name): self.wandb.Image(image)}, step=self.step
        )

    def add_audio(self, audio_name, audio, sample_rate=None):
        """
        Log an audio to the experiment tracker.

        Args:
            audio_name (str): name of the audio to use in the tracker.
            audio (torch.Tensor): audio to log.
            sample_rate (int): audio sample rate.
        """
        a = audio.detach().cpu()
        if a.ndim == 2 and a.shape[0] == 1:  # (1, T) -> (T,)
            a = a.squeeze(0)
        a = a.numpy()
        if a.ndim == 2 and a.shape[0] < a.shape[1]:  # (C, T) -> (T, C)
            a = a.T

        self.wandb.log(
            {
                self._object_name(audio_name): self.wandb.Audio(
                    a, sample_rate=sample_rate
                )
            },
            step=self.step,
        )

    def add_text(self, text_name, text):
        """
        Log text to the experiment tracker.

        Args:
            text_name (str): name of the text to use in the tracker.
            text (str): text content.
        """
        self.wandb.log(
            {self._object_name(text_name): self.wandb.Html(text)}, step=self.step
        )

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        """
        Log histogram to the experiment tracker.

        Args:
            hist_name (str): name of the histogram to use in the tracker.
            values_for_hist (Tensor): array of values to calculate
                histogram of.
            bins (int | None): the definition of bins for the histogram.
        """
        values_for_hist = values_for_hist.detach().cpu().numpy()
        if bins:
            np_hist = np.histogram(values_for_hist, bins=bins)
        else:
            np_hist = np.histogram(values_for_hist)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(values_for_hist, bins=512)

        hist = self.wandb.Histogram(np_histogram=np_hist)

        self.wandb.log({self._object_name(hist_name): hist}, step=self.step)

    def add_table(self, table_name, table: pd.DataFrame):
        """
        Log table to the experiment tracker.

        Args:
            table_name (str): name of the table to use in the tracker.
            table (DataFrame): table content.
        """
        self.wandb.log(
            {self._object_name(table_name): self.wandb.Table(dataframe=table)},
            step=self.step,
        )

    def add_images(self, image_names, images):
        raise NotImplementedError()

    def add_pr_curve(self, curve_name, curve):
        raise NotImplementedError()

    def add_embedding(self, embedding_name, embedding):
        raise NotImplementedError()
