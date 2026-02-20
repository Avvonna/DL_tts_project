# TTS Vocoder Project (HiFi-GAN)

Репозиторий содержит реализацию нейросетевого вокодера **HiFi-GAN** для синтеза речи на русском языке.
Проект выполнен в рамках ДЗ№3 Sound DL (TTS).

## Особенности реализации

* **Архитектура:** HiFi-GAN (Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis).
    * **Generator:** На основе Transposed Convs и Multi-Receptive Field Fusion (MRF).
    * **Discriminators:** Multi-Period Discriminator (MPD) + Multi-Scale Discriminator (MSD).
* **Лосс:** Комбинация Adversarial Loss, Feature Matching Loss и Mel-Spectrogram Reconstruction Loss.
* **Пайплайн:**
    * **Resynthesis:** Восстановление аудио из mel-спектрограмм (Ground Truth).
    * **Full TTS:** Генерация речи из текста (Text -> Acoustic Model -> Mel -> Vocoder). В качестве акустической модели используется **Facebook MMS (Massively Multilingual Speech)**.
* **Данные:** Обучение на датасете **RUSLAN** (22050 Hz).
* **Логирование:** WandB.

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/Avvonna/DL_tts_project.git
   cd DL_tts_project
   ```

2. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

3. **Загрузка весов модели:**
   Для запуска инференса необходимо скачать обученные веса генератора (`generator.pth`).

   Вы можете сделать это автоматически с помощью скрипта:

   ```bash
   python scripts/download_generator.py \
       --file_id "1c8udP8vkwpQ-D6K3OCKKYPvLTDCIDs28" \
       --output_dir "saved/"
   ```
   *Либо скачайте веса вручную и поместите их в папку `saved/`.*

## Инференс

Для синтеза аудио используется скрипт [**synthesize.py**](./synthesize.py). Конфигурация управляется через Hydra.

### 1. Подготовка данных

Данные для инференса должны быть организованы в структуру `CustomDirDataset`:

```text
my_dataset/
├── audio/            # Для режима Resynthesis (GT Audio)
│   ├── file1.wav
│   └── ...
└── transcriptions/   # Для режима Full TTS (Тексты)
    ├── file1.txt
    └── ...
```

**Важно:** Имена файлов в `audio/` и `transcriptions/` должны совпадать (например, `sample1.wav` и `sample1.txt`).

### 2. Режим Resynthesis (Audio -> Mel -> Vocoder)

Используется для оценки качества вокодера в изоляции. Мел-спектрограммы извлекаются из оригинального аудио.

```bash
python synthesize.py \
   synthesize.mode="resynthesis" \
   synthesize.checkpoint_path="saved/generator.pth" \
   dataset.data_dir="data/test_samples" \
   synthesize.output_dir="outputs/resynthesis"
```

### 3. Режим Full TTS (Text -> AM -> Mel -> Vocoder)

Полный цикл синтеза. Текст преобразуется в мел-спектрограмму с помощью предобученной акустической модели (HuggingFace MMS), а затем озвучивается вашим вокодером.

```bash
python synthesize.py \
   synthesize.mode="full_tts" \
   synthesize.checkpoint_path="saved/generator.pth" \
   dataset.data_dir="data/test_samples" \
   synthesize.output_dir="outputs/full_tts" \
   synthesize.acoustic_model.model_name="facebook/mms-tts-rus"
```

* Результаты будут сохранены в указанную папку (`outputs/full_tts`).
* Будут сохранены как результаты работы вокодера, так и промежуточные "черновые" аудио от акустической модели (для сравнения).

## Демонстрация
Для быстрой проверки работы модели и воспроизведения результатов доступен **Jupyter Notebook**:
`demo.ipynb`

Он позволяет:
1. Скачать необходимые веса.
2. Запустить вокодер на тестовых предложениях (MOS).
3. Сравнить Resynthesis и Full TTS режимы.
4. Визуализировать спектрограммы и прослушать аудио прямо в браузере.

[**Открыть в Google Colab**](https://colab.research.google.com/github/Avvonna/DL_tts_project/blob/main/demo_colab.ipynb)

## Отчет о проделанной работе

### 1. Обучение
Обучение проводилось на датасете RUSLAN (одноголосый мужской голос).
* **Конфигурация:** [train.yaml](./src/configs/train.yaml).
* **Параметры:** `sr=22050`, `n_fft=1024`, `n_mels=80`.
* **Sanity Check:** Проведен успешный тест на обучение на одном батче ([hifigan_onebatch.yaml](./src/configs/experiment/hifigan_onebatch.yaml))

Команда для запуска обучения:
```bash
python train.py writer.run_name="HiFiGAN_Ruslan"
```

### 2. Логи обучения
Полные логи обучения, графики Mel-Loss и метрики дискриминаторов доступны в WandB: [**LINK TO W&B REPORT**](https://wandb.ai/vaalkaev-hse/DL_tts_hw/reports/Untitled-Report--VmlldzoxNTk4MjQ3Ng)

**Динамика обучения:**
* Модель, используемая для инференса, обучалась 500 эпох.
* Использовался `MultiPeriodDiscriminator` (периоды 2, 3, 5, 7, 11) и `MultiScaleDiscriminator`.
* Scheduler: ExponentialLR с gamma 0.999.

### 3. Результаты и MOS

Оценка качества проводилась на тестовых предложениях из задания. Для удобства сравнения и прослушивания все результаты генерации собраны в папке [**MOS**](./MOS).

**Состав файлов:**

В папке находятся примеры для каждого из 3-х тестовых предложений:
* `1.wav`, `2.wav`, `3.wav` — **Ground Truth**. Оригинальные аудиозаписи.
* `*_resynthesis.wav` — **Resynthesis**. Результат восстановления аудио из мел-спектрограмм оригиналов с помощью обученного вокодера.
* `*_hf.wav` — **Acoustic Model Reference**. Аудио, сгенерированное предобученной акустической моделью (Facebook MMS) "как есть". Позволяет оценить качество самой акустической модели без влияния нашего вокодера.
* `*_full_tts.wav` — **Full TTS Pipeline**. Результат полного цикла: *Текст → Мел-спектрограмма (MMS) → **Наш Вокодер***.

| Mode | Audio Quality (MOS Estimate) | Artifacts |
| :--- | :--- | :--- |
| **Ground Truth** | 5.0 | - |
| **Resynthesis** | **[3.0 - 3.5]** | Звучание "как из бочки", заметный металлический оттенок, периодический неприятный треск (щелчки). |
| **Full TTS (MMS)** | **[2.0 - 3.0]** | Наследует артефакты вокодера. Дополнительные проблемы акустической модели: дублирование звуков (эффект заикания/растягивания), неестественные ударения. |

* **Resynthesis:** Вокодер справляется с генерацией речи, но присутствуют характерные для GAN артефакты (металлический звон) и проблемы с фазой/фильтрацией, дающие эффект "бочки".
* **Full TTS:** Использование сторонней акустической модели (`facebook/mms-tts-rus`) вносит дополнительные искажения в длительность фонем (растягивание), которые слышны уже на этапе генерации промежуточного `_hf.wav`.

Результаты анализа спектрограмм (последняя часть ДЗ) находятся в отчете WanDB, картинки в него были загружены из папки [**results_images**](./results_images/), получены скриптом [visualize_batch.py](./scripts/visualize_batch.py)


### 4. Эксперименты и Улучшения
В ходе работы были реализованы следующие компоненты:

1. **HiFi-GAN Generator:** Полная реализация архитектуры из статьи (V1).
2. **Discriminator Ensemble:** Реализованы MPD и MSD для захвата периодических и масштабных структур аудио.
3. **Loss Functions:**
   * `Feature Matching Loss`: для стабилизации обучения.
   * `Mel-Reconstruction Loss`: для ускорения сходимости контента.
4. **Full TTS Pipeline:** Интеграция с `transformers` для использования сторонней акустической модели (`facebook/mms-tts-rus`), что позволяет синтезировать речь из произвольного текста.
5. **Утилиты:**
   * Скрипт `scripts/extract_generator.py` для облегчения весов (удаление оптимизатора и дискриминаторов из чекпоинта).
   * Автоматическое скачивание весов через `gdown`.

## Структура проекта

```
.
├── data/                   # Датасеты и кэш
├── saved/                  # Чекпоинты и логи
├── scripts/                # Утилиты (загрузка/извлечение весов)
├── src/
│   ├── configs/            # Конфиги Hydra (model, train, synthesize...)
│   ├── datasets/           # Загрузка данных (RUSLAN, CustomDir)
│   ├── logger/             # WandB логгер
│   ├── loss/               # Generator & Discriminator Losses
│   ├── metrics/            # Метрики
│   ├── model/              # HiFi-GAN (Generator, MPD, MSD), AcousticModel wrapper
│   ├── trainer/            # GANTrainer (training loop)
│   ├── transforms/         # MelSpectrogram extraction
│   └── utils/              # Вспомогательные функции
├── train.py                # Скрипт обучения
├── synthesize.py           # Скрипт инференса
├── demo.ipynb              # Демонстрационный ноутбук
└── requirements.txt        # Зависимости
```
