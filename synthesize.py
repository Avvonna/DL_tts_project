import logging
import time
from pathlib import Path

import hydra
import soundfile as sf
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from src.datasets.collate import collate_fn
from src.model.acoustic_model import HF_AcousticModel
from src.transforms.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig
from src.utils.device import resolve_device
from src.utils.init_utils import configure_logging_for_inference
from src.utils.io_utils import abs_path

logger = logging.getLogger("synthesize")

def _extract_generator_state(ckpt: object) -> dict:
    """
    Возвращает state_dict генератора
    """
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]

        # отдельный под-словарь generator
        if "generator" in sd and isinstance(sd["generator"], dict):
            return sd["generator"]

        # плоский state_dict с префиксом generator
        if any(k.startswith("generator.") for k in sd.keys()):
            return {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}

        # state_dict генератора
        return sd

    if isinstance(ckpt, dict):
        return ckpt

    raise RuntimeError("Не удалось распознать формат чекпоинта генератора.")

def _convert_weightnorm_keys_to_parametrizations(gen_sd: dict) -> dict:
    """
    Конвертирует старые ключи weight_norm (weight_g/weight_v) в формат
    torch.nn.utils.parametrizations.weight_norm (original0/original1)
    """
    has_wg = any(k.endswith(".weight_g") for k in gen_sd.keys())
    has_wv = any(k.endswith(".weight_v") for k in gen_sd.keys())
    if not (has_wg or has_wv):
        return gen_sd

    out = {}
    for k, v in gen_sd.items():
        k2 = k
        k2 = k2.replace(".weight_g", ".parametrizations.weight.original0")
        k2 = k2.replace(".weight_v", ".parametrizations.weight.original1")
        out[k2] = v
    return out

@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(cfg: DictConfig) -> None:
    configure_logging_for_inference()

    device = resolve_device(cfg.synthesize.device)
    out_dir = abs_path(cfg.synthesize.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = cfg.synthesize.get("mode", "resynthesis")
    logger.info(f"Запуск в режиме: {mode.upper()}")

    # Загрузка HiFi-GAN
    logger.info("Загрузка HiFi-GAN...")
    ckpt_path = abs_path(cfg.synthesize.checkpoint_path)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    generator = instantiate(cfg.generator).to(device)

    ckpt_path = abs_path(cfg.synthesize.checkpoint_path)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    gen_state = _extract_generator_state(ckpt)
    convert_wn = bool(cfg.synthesize.get("convert_weightnorm_keys", False))
    if convert_wn:
        gen_state = _convert_weightnorm_keys_to_parametrizations(gen_state)

    generator.load_state_dict(gen_state, strict=True)

    if cfg.synthesize.get("remove_weight_norm", True) and hasattr(generator, "remove_weight_norm"):
        generator.remove_weight_norm()

    generator.eval()

    # Инициализация mel-трансформа
    mel_cfg = MelSpectrogramConfig(
        sr=cfg.dataset.target_sr,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        n_mels=cfg.dataset.n_mels,
        f_min=cfg.dataset.f_min,
        f_max=cfg.dataset.f_max
    )
    mel_transform = MelSpectrogram(mel_cfg).to(device)

    # Загрузка акустической модели
    acoustic_model = None
    if mode == "full_tts":
        model_name = cfg.synthesize.acoustic_model.model_name
        logger.info(f"Загрузка HuggingFace модели: {model_name}...")
        acoustic_model = HF_AcousticModel(model_name, device)

    # Датасет
    dataset = instantiate(cfg.dataset)
    loader = instantiate(cfg.dataloader, dataset=dataset, collate_fn=collate_fn, shuffle=False, drop_last=False)

    out_dir_cfg = cfg.synthesize.get("output_dir", "auto")
    if out_dir_cfg in (None, "auto", "dataset"):
        data_dir = getattr(dataset, "data_dir", cfg.dataset.get("data_dir", "data"))
        out_dir = abs_path(str(Path(data_dir) / "outputs"))
    else:
        out_dir = abs_path(str(out_dir_cfg))
    out_dir.mkdir(parents=True, exist_ok=True)

    target_sr = int(cfg.preprocess.audio.sr)
    save_hf_audio = bool(cfg.synthesize.get("save_hf_audio", True))

    total_time_inference = 0.0
    total_duration_gen = 0.0

    # Ресемплер для Full TTS
    resampler = None

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Synthesizing"):
            paths = batch.get("audio_path", [])
            texts = batch.get("text", [])

            mel_to_vocode = None

            if mode == "full_tts" and acoustic_model:
                if all((t is None) or (str(t).strip() == "") for t in texts):
                    raise RuntimeError("FULL_TTS: все тексты пустые. Проверьте папку transcriptions и соответствие имен *.txt к *.wav.")

                # текст -> HF Model -> Audio -> Mel
                mels_list = []
                for text in texts:
                    if not text:
                        mels_list.append(None)
                        continue

                    # Генерируем черновое аудио
                    wav_am, am_sr = acoustic_model.generate_audio(text) # (1, T_raw)

                    # Ресемплим под частоту HiFi-GAN
                    if am_sr != target_sr:
                        if resampler is None:
                            resampler = torchaudio.transforms.Resample(am_sr, target_sr).to(device)
                        wav_am = resampler(wav_am)

                    # Извлекаем Mel-спектрограмму
                    mel = mel_transform(wav_am).squeeze(0) # (n_mels, T)
                    mels_list.append(mel)

                    if save_hf_audio:
                        stem = f"sample_{batch_idx:06d}"
                        if isinstance(paths, list) and len(paths) > 0:
                            stem = Path(paths[0]).stem
                        hf_path = out_dir / f"{stem}_hf.wav"
                        sf.write(
                            str(hf_path),
                            wav_am.detach().cpu().squeeze().numpy(),
                            target_sr
                        )

                # Собираем батч
                if any(m is not None for m in mels_list):
                    mels_T = [m.transpose(0, 1) for m in mels_list if m is not None]
                    mel_padded = pad_sequence(mels_T, batch_first=True, padding_value=-11.5129)
                    mel_to_vocode = mel_padded.transpose(1, 2).to(device) # (B, n_mels, T)
                else:
                    continue

            else:
                # Resynthesis
                if "mel" not in batch:
                    continue
                mel_to_vocode = batch["mel"].to(device)

            # Инференс HiFi-GAN
            t0 = time.perf_counter()
            audio_fake = generator(mel_to_vocode) # (B, 1, T)
            inference_time = time.perf_counter() - t0

            bsz = int(audio_fake.size(0))
            batch_duration = (audio_fake.size(-1) / target_sr) * bsz

            total_duration_gen += batch_duration
            total_time_inference += inference_time

            audio_fake = audio_fake.detach().cpu()

            # Сохранение
            bsz = int(audio_fake.size(0))
            for i in range(bsz):
                stem = f"sample_{batch_idx:06d}_{i}"
                if i < len(paths):
                    stem = Path(paths[i]).stem

                gen_path = out_dir / f"{stem}_{mode}.wav"
                sf.write(str(gen_path), audio_fake[i].squeeze().numpy(), target_sr)

    # Логи
    if total_duration_gen > 0:
        rtf = total_time_inference / total_duration_gen
        logger.info(f"Режим: {mode}")
        logger.info(f"RTF: {rtf:.4f}")
        logger.info(f"Сохранено в: {out_dir}")

if __name__ == "__main__":
    main()
