import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]) -> dict:
    """
    Собирает список элементов датасета в один батч.

    Что ожидается в каждом item:
      - spectrogram: тензор спектрограммы размера [F, T] или [1, F, T]
      - audio_path (или path): путь до исходного аудио (нужно, чтобы потом понимать что за файл)

    Дополнительно (если есть):
      - spectrogram_length: длина по времени (если нет, берем из самой спектрограммы)
      - text: строка с транскрипцией
      - text_encoded: тензор токенов [L] или [1, L]
      - utt_id / id / path: любой идентификатор (просто прокидываем списком для инференса)

    На выходе:
      dict с padded спектрограммами и (если есть) текстами/идентификаторами.
    """
    batch: dict = {}

    # Спектрограммы: приводим к [F, T], паддим по времени
    spectrograms_t_first = []   # сюда складываем [T, F], так удобнее pad_sequence
    spectrogram_lengths = []    # реальные длины по времени (до паддинга)

    for item in dataset_items:
        spec = item["spectrogram"]

        # иногда спектрограмма приходит как [1, F, T], убираем канал
        if spec.dim() == 3:
            spec = spec.squeeze(0)

        # тут ожидаем ровно 2 измерения
        if spec.dim() != 2:
            raise ValueError(f"Spectrogram должен быть [F,T], а пришло {tuple(spec.shape)}")

        # pad_sequence паддит по первой оси, поэтому делаем [T, F]
        spec_tf = spec.transpose(0, 1)
        spectrograms_t_first.append(spec_tf)

        # если длина явно лежит в item, берем ее, иначе считаем по форме
        if "spectrogram_length" in item:
            spectrogram_lengths.append(int(item["spectrogram_length"]))
        else:
            spectrogram_lengths.append(spec_tf.shape[0])

    # паддим до максимальной длины по времени
    padded_tf = pad_sequence(spectrograms_t_first, batch_first=True, padding_value=0.0)  # [B, Tmax, F]
    batch["spectrogram"] = padded_tf.transpose(1, 2).contiguous()                        # [B, F, Tmax]
    batch["spectrogram_length"] = torch.tensor(spectrogram_lengths, dtype=torch.long)

    # Пути до аудио
    if all("audio_path" in it for it in dataset_items):
        batch["audio_path"] = [it["audio_path"] for it in dataset_items]
    elif all("path" in it for it in dataset_items):
        batch["audio_path"] = [it["path"] for it in dataset_items]

    # Идентификаторы (опционально): если есть, прокидываем как список
    for key in ("utt_id", "id", "path"):
        if all(key in it for it in dataset_items):
            batch[key] = [it[key] for it in dataset_items]
            break

    # Текст (опционально): если во всех item есть text, кладем списком
    if all("text" in it for it in dataset_items):
        batch["text"] = [it["text"] for it in dataset_items]

    # Закодированный текст (опционально): паддим по длине L
    if all("text_encoded" in it for it in dataset_items):
        encoded = []
        encoded_lens = []

        for it in dataset_items:
            te = it["text_encoded"]

            # иногда приходит [1, L], убираем лишнюю ось
            if te.dim() == 2:
                te = te.squeeze(0)

            if te.dim() != 1:
                raise ValueError(f"text_encoded должен быть [L], а пришло {tuple(te.shape)}")

            encoded.append(te)
            encoded_lens.append(te.shape[0])

        batch["text_encoded"] = pad_sequence(encoded, batch_first=True, padding_value=0).contiguous()
        batch["text_encoded_length"] = torch.tensor(encoded_lens, dtype=torch.long)

    return batch
