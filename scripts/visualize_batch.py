import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.model.hifigan import Generator  # noqa: E402
from src.transforms.mel_spectrogram import (  # noqa: E402
    MelSpectrogram,
    MelSpectrogramConfig,
)


def load_generator(checkpoint_path, device):
    """Загрузка обученного генератора"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    generator = Generator(
        in_channels=80,
        upsample_rates=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ).to(device)

    # Обработка разных форматов чекпоинтов
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        if "generator" in sd:
            sd = sd["generator"]

        new_sd = {}
        for k, v in sd.items():
            key = k[10:] if k.startswith("generator.") else k
            new_sd[key] = v
        generator.load_state_dict(new_sd)
    else:
        generator.load_state_dict(checkpoint)

    if hasattr(generator, "remove_weight_norm"):
        try:
            generator.remove_weight_norm()
        except Exception:
            pass
    generator.eval()
    return generator


def get_mel_transform(device):
    """Настройки Mel как в конфиге обучения"""
    cfg = MelSpectrogramConfig(
        sr=22050,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        f_min=0.0,
        f_max=8000.0,
        power=1.0,
    )
    return MelSpectrogram(cfg).to(device)


def plot_comparison(mel_gt, mel_fake, filename, save_dir, left_label):
    """
    Рисует две спектрограммы: GT/HF, Fake.
    """
    save_path = save_dir / f"{filename}_comparison.png"

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # GT
    im1 = axes[0].imshow(
        mel_gt, origin="lower", aspect="auto", cmap="inferno", vmin=-11.5, vmax=2.5
    )
    axes[0].set_title(f"{left_label}: {filename}")
    axes[0].set_ylabel("Mel Channels (80)")
    axes[0].set_xlabel("Time Frames")
    fig.colorbar(im1, ax=axes[0], format="%+2.0f dB")

    # Fake
    im2 = axes[1].imshow(
        mel_fake, origin="lower", aspect="auto", cmap="inferno", vmin=-11.5, vmax=2.5
    )
    axes[1].set_title(f"Vocoder Resynthesis: {filename}")
    axes[1].set_xlabel("Time Frames")
    fig.colorbar(im2, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Batch visualization of Mel Spectrograms (GT vs Vocoder)"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to generator checkpoint (.pth)"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing .wav files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualization_results",
        help="Directory to save images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--left_label",
        type=str,
        default="Ground Truth"
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # Создаем выходную папку
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    audio_files = list(input_dir.glob("*.wav"))

    if not audio_files:
        print(f"No .wav files found in {input_dir}")
        return

    print(f"Loading model on {device}...")
    generator = load_generator(args.ckpt, device)
    mel_transform = get_mel_transform(device)

    print(f"Found {len(audio_files)} files. Processing...")

    for audio_path in tqdm(audio_files):
        try:
            # Загрузка и препроцессинг аудио
            wav_np, sr = sf.read(str(audio_path))

            # Конвертация numpy -> torch tensor
            wav = torch.from_numpy(wav_np).float().to(device)

            # sf.read возвращает (Frames, Channels) или (Frames,), а нам нужно (Channels, Frames)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # (T) -> (1, T)
            else:
                wav = wav.transpose(0, 1)  # (T, C) -> (C, T)

            # Стерео в моно
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # Ресемплинг
            if sr != 22050:
                r = torchaudio.transforms.Resample(sr, 22050).to(device)
                wav = r(wav)

            # Получаем GT mel
            with torch.no_grad():
                mel_gt_tensor = mel_transform(wav)  # (1, n_mels, T)

            # Прогоняем через генератор (GT Mel -> Fake Audio)
            with torch.no_grad():
                audio_fake = generator(mel_gt_tensor)

            # Получаем fake mel
            with torch.no_grad():
                mel_fake_tensor = mel_transform(audio_fake.squeeze(1))  # (1, n_mels, T)

            # Конвертация в numpy
            mel_gt = mel_gt_tensor.squeeze().cpu().numpy()
            mel_fake = mel_fake_tensor.squeeze().cpu().numpy()

            # Обрезаем, чтобы длины совпадали
            min_len = min(mel_gt.shape[1], mel_fake.shape[1])
            mel_gt = mel_gt[:, :min_len]
            mel_fake = mel_fake[:, :min_len]

            plot_comparison(mel_gt, mel_fake, audio_path.stem, out_dir, args.left_label)

        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Done! Results saved to {out_dir}")

if __name__ == "__main__":
    main()
