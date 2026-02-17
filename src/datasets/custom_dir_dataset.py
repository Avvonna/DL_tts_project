import logging
from pathlib import Path
from typing import Any, Optional

from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomDirDataset(BaseDataset):
    """
    Dataset для инференса/валидации на кастомных данных.

    Ожидаемая структура папки:
    root_dir/
    ├── audio/
    │   ├── file1.wav
    │   └── ...
    └── transcriptions/
        ├── file1.txt
        └── ...
    """

    # Поддерживаемые форматы аудио
    AUDIO_EXTS = {".wav", ".flac", ".mp3"}

    def __init__(
        self,
        data_dir: str,
        audio_dir: Optional[str] = None,
        transcription_dir: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir (str): Путь к корневой директории датасета.
            audio_dir (str, optional): Переопределить путь к папке audio.
            transcription_dir (str, optional): Переопределить путь к папке transcriptions.
            *args, **kwargs: Аргументы для BaseDataset (target_sr, limit, compute_mel и т.д.).
        """
        root = Path(data_dir)

        # если явно не передали путь к аудио, используем root/audio
        self.audio_path = Path(audio_dir) if audio_dir else root / "audio"

        # если явно не передали путь к транскрипциям, используем root/transcriptions
        self.trans_path = (
            Path(transcription_dir) if transcription_dir else root / "transcriptions"
        )

        # строим индекс
        index = self._build_index()

        # BaseDataset возьмет этот индекс и будет сам грузить аудио и считать мелы
        super().__init__(index=index, *args, **kwargs)

    def _build_index(self) -> list[dict[str, Any]]:
        """
        Сканирует директории и создает список словарей для BaseDataset.
        """
        # Проверка наличия аудио
        if not self.audio_path.exists() or not self.audio_path.is_dir():
            raise FileNotFoundError(f"Папка с аудио не найдена: {self.audio_path}")

        # Проверка наличия транскрипций (не критично, только ворнинг)
        transcriptions_found = False
        if self.trans_path.exists() and self.trans_path.is_dir():
            transcriptions_found = True
        else:
            logger.info(
                f"Папка с транскрипциями '{self.trans_path}' не найдена. "
                "Датасет будет загружен без текстов (режим Resynthesis)."
            )

        # Собираем аудиофайлы
        audio_files = []
        for ext in self.AUDIO_EXTS:
            audio_files.extend(self.audio_path.glob(f"*{ext}"))
            audio_files.extend(self.audio_path.glob(f"*{ext.upper()}"))

        audio_files = sorted(list(set(audio_files)))  # Убираем дубли и сортируем

        if not audio_files:
            raise FileNotFoundError(
                f"В папке {self.audio_path} не найдено аудиофайлов."
            )

        data: list[dict[str, Any]] = []
        missing_texts = 0

        for audio_file in audio_files:
            utt_id = audio_file.stem
            text = ""

            # Если папка с текстом существует, пытаемся найти файл
            if transcriptions_found:
                txt_file = self.trans_path / f"{utt_id}.txt"
                if txt_file.exists():
                    try:
                        # Читаем текст, убираем переносы строк
                        text = txt_file.read_text(encoding="utf-8").strip()
                    except Exception as e:
                        logger.warning(f"Ошибка чтения текста для {utt_id}: {e}")
                else:
                    missing_texts += 1

            # Формируем запись для BaseDataset
            entry = {
                "path": str(audio_file.absolute()),
                "text": text,
                "utt_id": utt_id,
            }
            data.append(entry)

        # Логируем статистику
        logger.info(f"Найдено {len(data)} аудиофайлов в {self.audio_path}")

        if transcriptions_found:
            if missing_texts > 0:
                logger.warning(f"Отсутствует текст для {missing_texts} файлов.")
            else:
                logger.info("Для всех аудиофайлов найдены транскрипции.")

        return data
