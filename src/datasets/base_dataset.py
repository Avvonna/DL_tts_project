import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.transforms.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for TTS datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.

    Supported index fields per item:
    - path: str (required)
    - text: str (optional, can be absent for inference)
    - audio_len: float seconds (optional, will be computed if absent and needed)
    """

    def __init__(
        self,
        index: list[dict[str, Any]],
        target_sr: int = 22050,
        limit: Optional[int] = None,
        max_audio_length: Optional[float] = None,
        min_audio_length: Optional[float] = None,
        shuffle_index: bool = False,
        random_crop: bool = False,
        cache_dir: Optional[str] = None,
        # Параметры для mel-спектрограммы
        compute_mel: bool = False,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        segment_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            index (list[dict]): List containing dict for each element of the dataset.
                Must contain 'path'.
            target_sr (int): Target sample rate for audio.
            limit (int | None): If not None, limit the total number of elements.
            max_audio_length (float | None): Maximum allowed audio length in seconds.
            min_audio_length (float | None): Minimum allowed audio length in seconds.
            shuffle_index (bool): If True, shuffle the index (seed 42).
            cache_dir (str | Path | None): Directory to store audio lengths cache.
            compute_mel (bool): If True, compute and return mel spectrogram.
            n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins.
            hop_length (int): Length of hop between STFT windows.
            n_mels (int): Number of mel filterbanks.
            f_min (float): Minimum frequency.
            f_max (float): Maximum frequency.
            segment_size (int | None): Length of the returned segment (in samples)
        """
        self.target_sr = int(target_sr)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

        self.segment_size = segment_size
        self.shuffle_index = bool(shuffle_index)
        self.random_crop = bool(random_crop)

        # Mel transform
        self.mel_transform = None
        if compute_mel:
            cfg = MelSpectrogramConfig(
                sr=self.target_sr,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                power=1.0,
                eps=1e-5,
                mel_scale="slaney",
                norm="slaney",
            )
            self.mel_transform = MelSpectrogram(cfg)

        self._assert_index_is_valid(index)

        # Подготовка индекса
        prepared_index = self.add_audio_lengths(index, cache_dir=self.cache_dir)
        prepared_index = self.filter_by_audio_len(
            prepared_index,
            min_len=min_audio_length,
            max_len=max_audio_length,
        )

        # shuffle/limit после фильтрации
        prepared_index = self.shuffle_and_limit(
            prepared_index,
            limit=limit,
            shuffle_index=self.shuffle_index,
            seed=42,
        )

        self._index: list[dict[str, Any]] = prepared_index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, ind: int) -> dict[str, Any]:
        data = self._index[ind]
        audio_path = data["path"]

        # Загружаем полное аудио
        audio = self.load_audio(audio_path)  # (1, T)

        # Кроппинг/паддинг аудио
        if self.segment_size is not None:
            audio = self._crop_or_pad_audio(audio)  # (1, target_T)

        # Вычисляем mel из обработанного аудио
        mel = None
        if self.mel_transform is not None:
            # audio: (1, T) -> mel_transform вернет (1, n_mels, T_mel)
            mel = self.mel_transform(audio).squeeze(0)  # (n_mels, T_mel)

        result = {
            "audio": audio,
            "audio_path": audio_path,
            "segment_duration": audio.size(-1) / self.target_sr,
            **{k: v for k, v in data.items() if k not in {"path", "audio_len"}},
        }

        if mel is not None:
            result["mel"] = mel

        return result

    def _crop_or_pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Режет или паддит аудио до segment_size.
        audio: (1, T)
        returns: (1, segment_size)
        """
        if self.segment_size is None:
            return audio

        T = int(audio.size(-1))
        target_T = int(self.segment_size)

        # Если короче - паддим
        if T < target_T:
            padding_size = target_T - T
            audio = F.pad(audio, (0, padding_size), mode="constant", value=0.0)
            return audio

        # Если длиннее - режем
        if T > target_T:
            if self.random_crop:  # train: случайный кроп
                start = random.randint(0, T - target_T)
            else:  # val: начало
                start = 0
            audio = audio[:, start : start + target_T]

        return audio

    def load_audio(self, path: str) -> torch.Tensor:
        try:
            audio_np, sr = sf.read(
                path,
                dtype="float32",
                always_2d=True,
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки аудио: {path}") from e

        audio = torch.from_numpy(audio_np).transpose(0, 1)  # (C, T)

        # Конвертация в mono
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Ресэмплинг (с кэшированием ресэмплеров)
        if sr != self.target_sr:
            sr = int(sr)
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.target_sr,
                )
            audio = self._resamplers[sr](audio)

        # Ограничение амплитуды
        peak = float(audio.abs().max())
        if peak > 1.0:
            audio = audio / peak

        return audio

    @staticmethod
    def filter_by_audio_len(
        index: list[dict[str, Any]],
        min_len: Optional[float],
        max_len: Optional[float],
    ) -> list[dict[str, Any]]:
        """
        Фильтрует элементы индекса по длительности аудио.
        """
        if not index:
            return index

        if not all("audio_len" in item for item in index):
            logger.warning(
                "Поле `audio_len` присутствует не у всех элементов. "
                "Фильтрация по длине пропущена."
            )
            return index

        initial_size = len(index)
        filtered: list[dict[str, Any]] = []

        for item in index:
            length = float(item["audio_len"])

            if min_len is not None and length < float(min_len):
                continue
            if max_len is not None and length > float(max_len):
                continue

            filtered.append(item)

        if len(filtered) != initial_size:
            logger.info(
                f"Отфильтровано {initial_size - len(filtered)} из "
                f"{initial_size} элементов по длительности."
            )

        return filtered

    def _assert_index_is_valid(self, index: list[dict[str, Any]]) -> None:
        """Проверяет корректность индекса датасета."""
        for i, entry in enumerate(index):
            if "path" not in entry:
                raise ValueError(
                    f"Элемент индекса #{i} не содержит обязательное поле `path`."
                )

    def _cache_path(self) -> Optional[Path]:
        """Возвращает путь к файлу кэша."""
        if self.cache_dir is None:
            return None

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / "audio_lengths_cache.json"

    @classmethod
    def add_audio_lengths(
        cls,
        index: list[dict[str, Any]],
        cache_dir: Optional[Path] = None,
    ) -> list[dict[str, Any]]:
        """Гарантирует наличие поля `audio_len` (секунды) у каждого элемента."""
        if index and all("audio_len" in item for item in index):
            return list(index)

        cache_path: Optional[Path] = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "audio_lengths_cache.json"

        # Загружаем кэш
        cache: dict[str, float] = {}
        if cache_path is not None and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                cache = {str(k): float(v) for k, v in raw.items()}
            except Exception as e:
                logger.warning(f"Не удалось загрузить кэш: {e}")
                cache = {}

        cache_updated = False
        logger.info("Вычисление длительности аудиофайлов...")

        result: list[dict[str, Any]] = []

        for item in tqdm(index, desc="Сканирование аудио"):
            new_item = dict(item)

            if "audio_len" in new_item:
                new_item["audio_len"] = float(new_item["audio_len"])
                result.append(new_item)
                continue

            path = str(new_item["path"])

            # Проверяем кэш
            if path in cache:
                new_item["audio_len"] = float(cache[path])
                result.append(new_item)
                continue

            # Считаем длительность
            try:
                info = sf.info(path)

                if info.frames == 0:
                    logger.warning(f"Пустой аудиофайл пропущен: {path}")
                    continue

                dur = float(info.duration)
                new_item["audio_len"] = dur
                cache[path] = dur
                cache_updated = True
                result.append(new_item)

            except Exception as e:
                logger.warning(f"Ошибка чтения аудиофайла {path}: {e}. Пропуск.")

        # Сохраняем кэш если были изменения
        if cache_updated and cache_path is not None:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Не удалось сохранить кэш: {e}")

        return result

    @staticmethod
    def shuffle_and_limit(
        index: list[dict[str, Any]],
        limit: Optional[int],
        shuffle_index: bool,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        index = list(index)

        if shuffle_index:
            random.seed(int(seed))
            random.shuffle(index)

        if limit is not None:
            index = index[: int(limit)]

        return index
