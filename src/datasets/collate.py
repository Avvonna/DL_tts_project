import torch
from torch.nn.utils.rnn import pad_sequence

MIN_MEL_VALUE = -11.5129251  # log(1e-5)


def collate_fn(dataset_items: list[dict]) -> dict:
    """
    Collate function для HiFi-GAN.

    Ожидается в каждом item:
      - mel: mel-спектрограмма [n_mels, T]
      - audio: waveform [1, T_audio] или [T_audio]

    На выходе:
      dict с padded mel и audio
    """
    # Списки для накопления данных
    mels_t_first = []
    mel_lengths = []
    audios = []
    audio_lengths = []
    paths = []
    texts = []
    utt_ids = []

    # Флаг наличия Mel
    has_mel = len(dataset_items) > 0 and "mel" in dataset_items[0]

    for item in dataset_items:
        # Обработка Audio
        audio = item["audio"]
        if audio.dim() == 2:
            audio = audio.squeeze(0)  # (1, T) -> (T)
        audios.append(audio)
        audio_lengths.append(audio.shape[0])

        # Обработка Mel
        if has_mel:
            mel = item["mel"]
            if mel.dim() == 2:
                # (n_mels, T) -> (T, n_mels) для pad_sequence
                mel = mel.transpose(0, 1)
            mels_t_first.append(mel)
            mel_lengths.append(mel.shape[0])

        paths.append(item.get("audio_path", ""))
        texts.append(item.get("text", ""))
        utt_ids.append(item.get("utt_id", ""))

    batch = {}

    # Паддинг Audio
    padded_audio = pad_sequence(
        audios, batch_first=True, padding_value=0.0
    )  # Выход (Batch, T_max)
    batch["audio"] = padded_audio.unsqueeze(1)  # Возвращаем канал (Batch, 1, T_max)
    batch["audio_length"] = torch.tensor(audio_lengths, dtype=torch.long)

    # Паддинг Mel
    if has_mel:
        padded_mel = pad_sequence(
            mels_t_first, batch_first=True, padding_value=MIN_MEL_VALUE
        )  # Выход (Batch, T_max, n_mels)
        batch["mel"] = padded_mel.transpose(
            1, 2
        ).contiguous()  # Разворот (Batch, n_mels, T_max)
        batch["mel_length"] = torch.tensor(mel_lengths, dtype=torch.long)

    batch["text"] = texts
    batch["audio_path"] = paths
    batch["utt_id"] = utt_ids

    return batch
