import argparse
import sys
from pathlib import Path

import torch


def extract_generator(input_path: str, output_path: str):
    """
    Функция, которая достает и сохраняет веса генератора из модели.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"Ошибка: Файл {input_file} не найден.")
        sys.exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка чекпоинта: {input_file} ...")

    # Загружаем на CPU
    try:
        checkpoint = torch.load(input_file, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        sys.exit(1)

    if not isinstance(checkpoint, dict):
        print("Ошибка: Чекпоинт не является словарем.")
        sys.exit(1)

    if "state_dict" not in checkpoint:
        print("Ошибка: Ключ 'state_dict' не найден в чекпоинте.")
        print(f"Доступные ключи: {list(checkpoint.keys())}")
        sys.exit(1)

    models_dict = checkpoint["state_dict"]

    if "generator" not in models_dict:
        print("Ошибка: Ключ 'generator' не найден внутри 'state_dict'.")
        print(f"Доступные модели внутри state_dict: {list(models_dict.keys())}")
        sys.exit(1)

    generator_weights = models_dict["generator"]

    print(f"Извлечение весов генератора ({len(generator_weights)} ключей)...")

    # Сохраняем веса генератора
    torch.save(generator_weights, output_file)

    print(f"Успешно сохранено в: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Извлечение весов генератора из полного чекпоинта обучения."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Путь к исходному .pth файлу (чекпоинту).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Путь для сохранения .pth файла только с генератором.",
    )

    args = parser.parse_args()

    extract_generator(args.input, args.output)
