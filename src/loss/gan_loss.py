import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.audio import match_audio_length


class DiscriminatorLoss(nn.Module):
    """
    Loss для обучения дискриминатора в HiFi-GAN.
    Использует MSE loss для GAN обучения.
    """

    def __init__(self):
        super().__init__()

    def forward(self, disc_outputs_real, disc_outputs_fake):
        """
        Args:
            disc_outputs_real: dict[str, list[(output, fmap)]]
                Выходы дискриминаторов на реальном аудио
            disc_outputs_fake: dict[str, list[(output, fmap)]]
                Выходы дискриминаторов на фейковом аудио

        Returns:
            loss: скаляр
            losses_dict: словарь с детализацией лоссов
        """
        loss = 0.0
        num_sub = 0
        losses_dict = {}

        for name in disc_outputs_real.keys():
            # disc_outputs_real[name] - это список (output, fmap) для каждого sub-discriminator
            real_outputs = disc_outputs_real[name]
            fake_outputs = disc_outputs_fake[name]

            for i, (real_out_fmap, fake_out_fmap) in enumerate(
                zip(real_outputs, fake_outputs)
            ):
                real_out = real_out_fmap[0]  # берем только output, без feature maps
                fake_out = fake_out_fmap[0]

                # MSE Loss: Real -> 1, Fake -> 0
                loss_real = F.mse_loss(real_out, torch.ones_like(real_out))
                loss_fake = F.mse_loss(fake_out, torch.zeros_like(fake_out))

                disc_loss = loss_real + loss_fake
                loss += disc_loss
                num_sub += 1

                losses_dict[f"{name}_{i}_real"] = loss_real
                losses_dict[f"{name}_{i}_fake"] = loss_fake

        loss = loss / max(num_sub, 1)
        return loss, losses_dict


class GeneratorLoss(nn.Module):
    """
    Loss для обучения генератора в HiFi-GAN.
    Состоит из:
    - Adversarial loss (обманываем дискриминатор)
    - Feature matching loss (совпадение промежуточных представлений)
    - Mel-spectrogram reconstruction loss
    """

    def __init__(self, lambda_fm=2.0, lambda_mel=45.0):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

    def forward(
        self,
        audio,
        audio_fake,
        disc_outputs_real,
        disc_outputs_fake,
        mel,
        mel_transform,
    ):
        """
        Args:
            audio: реальное аудио (не используется, нужен для совместимости)
            audio_fake: сгенерированное аудио
            disc_outputs_real: выходы D на реальном аудио
            disc_outputs_fake: выходы D на фейковом аудио
            mel: mel-спектрограмма (для reconstruction loss)

        Returns:
            loss: скаляр
            losses_dict: словарь с детализацией лоссов
        """
        loss_adv = 0.0
        loss_fm = 0.0
        n_adv = 0
        n_fm = 0
        losses_dict = {}

        for name in disc_outputs_fake.keys():
            real_outputs = disc_outputs_real[name]
            fake_outputs = disc_outputs_fake[name]

            for i, (real_out_fmap, fake_out_fmap) in enumerate(
                zip(real_outputs, fake_outputs)
            ):
                fake_out = fake_out_fmap[0]
                real_fmaps = real_out_fmap[1]
                fake_fmaps = fake_out_fmap[1]

                # Adversarial loss: generator хочет чтобы D думал что это real (=1)
                adv = F.mse_loss(fake_out, torch.ones_like(fake_out))
                loss_adv += adv
                n_adv += 1

                # Feature Matching loss: совпадение промежуточных feature maps
                for real_fm, fake_fm in zip(real_fmaps, fake_fmaps):
                    loss_fm += F.l1_loss(fake_fm, real_fm)
                    n_fm += 1

        loss_adv = loss_adv / max(n_adv, 1)
        loss_fm = loss_fm / max(n_fm, 1)

        # Mel-Spectrogram Reconstruction Loss
        mel_fake = mel_transform(audio_fake.squeeze(1))  # log-mel

        # Обработка несовпадения длин (из-за паддинга в STFT)
        mel_fake, mel_target = match_audio_length(mel_fake, mel)

        loss_mel = F.l1_loss(mel_fake, mel_target)

        # Итоговая сумма
        loss = loss_adv + self.lambda_fm * loss_fm + self.lambda_mel * loss_mel

        losses_dict["adv"] = loss_adv.item() if torch.is_tensor(loss_adv) else loss_adv
        losses_dict["fm"] = loss_fm.item() if torch.is_tensor(loss_fm) else loss_fm
        losses_dict["mel"] = loss_mel.item()

        return loss, losses_dict
