from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    """
    Dataset для инференса/валидации на кастомных данных.

    Ожидаемая структура папки:

    root_dir
    ├── audio
    │   ├── UtteranceID1.wav (или .flac/.mp3/.m4a)
    │   └── ...
    └── transcriptions (необязательно)
        ├── UtteranceID1.txt
        └── ...

    Идея:
      - берем все аудиофайлы из папки audio
      - для каждого аудио пытаемся найти txt с таким же именем в transcriptions
      - если require_text=True, то оставляем только те файлы, где txt найден
      - если require_text=False, то текст не обязателен (тогда метрики обычно не считаются)
    """

    AUDIO_EXTS = {".wav", ".flac", ".mp3"}

    def __init__(
        self,
        data_dir: str,
        audio_dir: Optional[str] = None,
        transcription_dir: Optional[str] = None,
        require_text: Optional[bool] = None,
        *args,
        **kwargs,
    ):
        root = Path(data_dir)

        # если явно не передали путь к аудио, используем root/audio
        if audio_dir is None:
            audio_path = root / "audio"
        else:
            audio_path = Path(audio_dir)

        # если явно не передали путь к транскрипциям, используем root/transcriptions
        if transcription_dir is None:
            trans_path = root / "transcriptions"
        else:
            trans_path = Path(transcription_dir)

        # проверяем, существует ли папка с транскрипциями
        has_trans_dir = trans_path.exists() and trans_path.is_dir()
        if require_text is None:
            require_text = bool(has_trans_dir)

        # строим индекс
        index = self._build_index(
            audio_path=audio_path,
            trans_path=trans_path if has_trans_dir else None,
            require_text=bool(require_text),
        )

        # BaseDataset дальше уже:
        # - (опционально) посчитает audio_len
        # - применит фильтры по длине (если заданы)
        # - применит transforms и т.д.
        super().__init__(
            index=index,
            require_text=bool(require_text),
            *args,
            **kwargs,
        )

    def _build_index(
        self,
        audio_path: Path,
        trans_path: Optional[Path],
        require_text: bool,
    ) -> List[Dict[str, Any]]:
        if not audio_path.exists() or not audio_path.is_dir():
            raise FileNotFoundError(f"audio directory does not exist: {audio_path}")

        # все файлы списком
        audio_files = sorted(
            [
                p
                for p in audio_path.iterdir()
                if p.is_file() and p.suffix.lower() in self.AUDIO_EXTS
            ]
        )

        data: List[Dict[str, Any]] = []
        for p in audio_files:
            # имя файла без расширения
            utt_id = p.stem

            # путь до аудио
            entry: Dict[str, Any] = {
                "path": str(p.absolute().resolve()),
                "utt_id": utt_id,
            }

            # если есть папка с транскрипциями, пытаемся подцепить текст
            if trans_path is not None:
                txt = trans_path / f"{utt_id}.txt"
                if txt.exists():
                    entry["text"] = txt.read_text(encoding="utf-8").strip()

            # если текст обязателен, то пропускаем файлы без транскрипции
            if require_text and "text" not in entry:
                continue

            data.append(entry)

        return data
