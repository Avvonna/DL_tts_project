import argparse
import os

import gdown


def download_weights(file_id: str, output_dir: str, filename: str):
    """
    Скачивает файл с GD в указанную директорию.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"Файл уже существует: {output_path}")
    else:
        print(f"Скачивание весов в {output_path}...")
        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, output_path, quiet=False, fuzzy=True)

        print("Веса успешно скачаны.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Скачивание предобученных весов с Google Drive."
    )

    parser.add_argument(
        "--file_id", type=str, required=True, help="ID файла на Google Drive"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Директория для сохранения файла (по умолчанию - корень директории)",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="generator.pth",
        help="Имя сохраняемого файла (по умолчанию: generator.pth)",
    )

    args = parser.parse_args()

    download_weights(args.file_id, args.output_dir, args.filename)
