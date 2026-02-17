from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class RUSLANDataset(BaseDataset):
    """
    Dataset для RUSLAN корпуса.

    Структура:
    root_dir
    ├── 000000_RUSLAN.wav
    ├── 000001_RUSLAN.wav
    ├── ...
    └── metadata_RUSLAN_22200.csv

    Формат CSV (без заголовка):
    000000_RUSLAN|С тревожным чувством берусь я за перо.
    000001_RUSLAN|Кого интересуют признания литературного неудачника?
    ...
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        split_seed: int = 42,
        val_size: int | float = 0.02,
        test_size: int | float = 0.0,
        metadata_file: Optional[str] = None,
        audio_extension: str = ".wav",
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir:
                Путь к директории RUSLAN с wav файлами и metadata.
            split:
                Какая часть датасета нужна: "train" | "val" | "test".
            split_seed:
                Seed для воспроизводимого разбиения.
            val_size:
                Размер val (доля 0..1 или число).
            test_size:
                Размер test (доля 0..1 или число). Если 0, тестовый сплит не выделяется.
            metadata_file:
                Имя файла с метаданными (по умолчанию: metadata_RUSLAN_22200.csv).
            audio_extension:
                Расширение аудиофайлов (по умолчанию: .wav).
            *args, **kwargs:
                Передаются в BaseDataset (target_sr, limit, shuffle_index, segment_size и т.д.).
        """
        root = Path(data_dir)

        # Определяем путь к файлу метаданных
        if metadata_file is None:
            metadata_path = root / "metadata_RUSLAN_22200.csv"
        else:
            metadata_path = root / metadata_file

        # Проверяем существование файла метаданных
        if not metadata_path.exists():
            raise FileNotFoundError(f"Файл метаданных не найден: {metadata_path}")

        split = str(split).lower().strip()
        if split not in {"train", "val", "test"}:
            raise ValueError("split должен быть одним из: train, val, test")

        # Строим полный индекс (до фильтрации и разбиения)
        full_index = self._build_index(
            root_dir=root,
            metadata_path=metadata_path,
            audio_extension=audio_extension,
        )

        # split делаем по фильтрованному индексу
        cache_dir = kwargs.get("cache_dir")
        cache_dir_path = Path(cache_dir) if cache_dir else None

        min_audio_length = kwargs.get("min_audio_length")
        max_audio_length = kwargs.get("max_audio_length")

        filtered = BaseDataset.add_audio_lengths(full_index, cache_dir=cache_dir_path)
        filtered = BaseDataset.filter_by_audio_len(
            filtered,
            min_len=min_audio_length,
            max_len=max_audio_length,
        )

        # Разбиение делаем до shuffle_index и до limit
        split_index = self._split_index(
            filtered,
            split=split,
            seed=int(split_seed),
            val_size=val_size,
            test_size=test_size,
        )

        # Выключаем фильтрацию для BaseDataset
        kwargs = dict(kwargs)
        kwargs["min_audio_length"] = None
        kwargs["max_audio_length"] = None

        super().__init__(
            index=split_index,
            compute_mel=True,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            *args,
            **kwargs,
        )

    def _build_index(
        self,
        root_dir: Path,
        metadata_path: Path,
        audio_extension: str,
    ) -> list[dict[str, Any]]:
        """
        Строит индекс датасета из CSV файла.

        Args:
            root_dir:
                Корневая директория с аудиофайлами.
            metadata_path:
                Путь к CSV файлу с метаданными.
            audio_extension:
                Расширение аудиофайлов.

        Returns:
            Список словарей с полями:
                - path: путь к аудиофайлу
                - text: текст транскрипции
                - utt_id: идентификатор utterance
        """
        try:
            df = pd.read_csv(
                metadata_path,
                sep="|",
                header=None,
                names=["utt_id", "text"],
                encoding="utf-8",
                dtype=str,
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения файла метаданных {metadata_path}: {e}") from e

        initial_size = len(df)

        df = df.dropna()
        df["text"] = df["text"].str.strip()
        df = df[df["text"] != ""]

        # Формируем путь к аудио и проверяем существование
        df["audio_path"] = df["utt_id"].apply(lambda x: root_dir / (x + audio_extension))
        df["exists"] = df["audio_path"].apply(lambda x: x.exists())
        df = df[df["exists"]]

        # Фиксируем порядок
        df = df.sort_values("utt_id").reset_index(drop=True)

        skipped = initial_size - len(df)
        if skipped > 0:
            logger.info(
                f"Пропущено {skipped} из {initial_size} записей "
                f"(отсутствующие файлы, пустые тексты или некорректные строки)."
            )

        if len(df) == 0:
            raise ValueError(
                f"Не найдено ни одной валидной записи в {metadata_path}. "
                f"Проверьте пути к аудиофайлам и формат CSV."
            )

        # Формируем список словарей
        data: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            entry: dict[str, Any] = {
                "path": str(row["audio_path"]),
                "text": row["text"],
                "utt_id": row["utt_id"],
            }
            data.append(entry)

        return data

    @staticmethod
    def _split_index(
        index: list[dict[str, Any]],
        split: str,
        seed: int,
        val_size: int | float,
        test_size: int | float,
    ) -> list[dict[str, Any]]:
        """
        Делит индекс на train/val/test без сохранения файлов со сплитом.

        split делается на уровне списка записей:
        - перемешиваем копию index фиксированным seed
        - берем первые val_count, затем test_count, остаток в train
        """
        n = len(index)
        if n == 0:
            logger.warning("После фильтрации индекс пустой. split вернет пустой датасет.")
            return []

        def to_count(x: int | float, total: int) -> int:
            # x <= 0: отключено
            if isinstance(x, (int, float)) and float(x) <= 0.0:
                return 0

            # 0 < x <= 1: доля
            if isinstance(x, float) and 0.0 < x <= 1.0:
                c = int(total * x)
                # если доля > 0, но округлилось в 0, делаем минимум 1
                return 1 if c == 0 and total > 1 else c

            try:
                return int(x)
            except Exception:
                return 0

        desired_val = to_count(val_size, n)
        desired_test = to_count(test_size, n)
        val_count = max(desired_val, 0)
        test_count = max(desired_test, 0)

        # Ограничиваем val/test по верхней границе n
        if val_count > n:
            logger.warning(f"val_size={desired_val} больше размера данных n={n}. val будет урезан до {n}.")
            val_count = n
        if test_count > n:
            logger.warning(f"test_size={desired_test} больше размера данных n={n}. test будет урезан до {n}.")
            test_count = n

        # Контролируем суммарный размер
        if val_count + test_count > n:
            logger.warning(
                f"val_size+test_size={val_count + test_count} больше n={n}. "
                f"test будет урезан до {max(n - val_count, 0)}."
            )
            test_count = max(n - val_count, 0)

        # Гарантия ненулевого train
        train_count = n - val_count - test_count
        if train_count <= 0 and n >= 2:
            need = 1 - train_count
            if val_count > 0:
                dec = min(val_count, need)
                val_count -= dec
                need -= dec
            if need > 0 and test_count > 0:
                dec = min(test_count, need)
                test_count -= dec
                need -= dec

            train_count = n - val_count - test_count
            logger.warning(
                f"После коррекции размеров выделено: train={train_count}, val={val_count}, test={test_count} (n={n})."
            )

        # Разбиение
        rng = random.Random(int(seed))
        shuffled = list(index)
        rng.shuffle(shuffled)

        val_part = shuffled[:val_count]
        test_part = shuffled[val_count : val_count + test_count]
        train_part = shuffled[val_count + test_count :]

        if split == "train":
            return train_part
        if split == "val":
            return val_part
        return test_part
