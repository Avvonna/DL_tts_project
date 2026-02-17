import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


def init_weights(m):
    """Универсальная функция инициализации весов."""
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
        # weight_norm (parametrizations)
        if hasattr(m, 'parametrizations'):
            weight_params = getattr(m.parametrizations, 'weight', None)
            if weight_params is not None:
                # original1 - вектор направления
                original1 = getattr(weight_params, 'original1', None)
                if original1 is not None and isinstance(original1, torch.Tensor):
                    nn.init.normal_(original1.data, mean=0.0, std=0.01)
                    return

        # spectral norm (weight_orig)
        weight_orig = getattr(m, 'weight_orig', None)
        if weight_orig is not None and isinstance(weight_orig, torch.Tensor):
            nn.init.normal_(weight_orig.data, mean=0.0, std=0.01)
            return

        # обычная инициализация весов (weight)
        weight = getattr(m, 'weight', None)
        if weight is not None and isinstance(weight, torch.Tensor):
            nn.init.normal_(weight.data, mean=0.0, std=0.01)


class ResBlock(nn.Module):
    """
    Residual блок с dilated convolutions.
    Используется в генераторе HiFi-GAN.
    """
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()

        # Несколько веток с разными dilation rates
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size,
                dilation=d, padding=self._get_padding(kernel_size, d)
            ))
            for d in dilation
        ])

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size,
                dilation=1, padding=self._get_padding(kernel_size, 1)
            ))
            for _ in dilation
        ])

        # Веса инициализируются в генераторе

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = x + xt  # residual connection
        return x

    def remove_weight_norm(self):
        for conv in self.convs1:
            remove_parametrizations(conv, "weight")
        for conv in self.convs2:
            remove_parametrizations(conv, "weight")

    @staticmethod
    def _get_padding(kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2


class Generator(nn.Module):
    """
    HiFi-GAN Generator.
    Преобразует mel-спектрограмму в аудио через transposed convolutions.
    """
    def __init__(
        self,
        in_channels=80,  # mel bins
        upsample_rates=(8, 8, 2, 2),  # stride для каждого upsample слоя
        upsample_kernel_sizes=(16, 16, 4, 4),
        upsample_initial_channel=512,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Входная conv для преобразования mel в скрытое представление
        self.conv_pre = weight_norm(nn.Conv1d(
            in_channels, upsample_initial_channel, 7, padding=3
        ))

        # Upsample блоки
        self.ups = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                upsample_initial_channel // (2 ** i),
                upsample_initial_channel // (2 ** (i + 1)),
                kernel, stride=rate, padding=(kernel - rate) // 2
            )))

        # Residual блоки после каждого upsample
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        # Выходная conv для получения waveform
        out_channels = upsample_initial_channel // (2 ** len(upsample_rates))
        self.conv_post = weight_norm(nn.Conv1d(out_channels, 1, 7, padding=3))

        # Инициализация весов
        self.apply(init_weights)

    def forward(self, x):
        """
        x: (B, mel_bins, T) - mel-spectrogram
        return: (B, 1, T * prod(upsample_rates)) - audio waveform
        """
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            # Суммируем выходы нескольких residual блоков
            xs = torch.zeros_like(x)
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)  # выход в [-1, 1]

        return x

    def remove_weight_norm(self):
        remove_parametrizations(self.conv_pre, "weight")
        for up in self.ups:
            remove_parametrizations(up, "weight")
        for block in self.resblocks:
            if isinstance(block, ResBlock):
                block.remove_weight_norm()
        remove_parametrizations(self.conv_post, "weight")


class PeriodDiscriminator(nn.Module):
    """
    Discriminator для одного периода.
    Reshape входа в 2D и применяет 2D convolutions.
    """
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period

        # Набор 2D convolutions
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])

        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

        # Инициализация весов
        self.apply(init_weights)

    def forward(self, x):
        """
        x: (B, 1, T)
        return: output, feature_maps
        """
        fmap = []

        # Reshape в 2D: разбиваем по периодам
        b, c, t = x.shape
        if t % self.period != 0:
            # Pad до кратного периоду
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """
    MPD - несколько Period Discriminators с разными периодами.
    """
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, x):
        """
        x: (B, 1, T)
        return: list of (output, feature_maps) для каждого дискриминатора
        """
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
        return outputs


class ScaleDiscriminator(nn.Module):
    """
    Один Scale Discriminator - стандартный 1D CNN.
    """
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

        # Инициализация весов
        self.apply(init_weights)

    def forward(self, x):
        """
        x: (B, 1, T)
        return: output, feature_maps
        """
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """
    MSD - три Scale Discriminators на разных разрешениях.
    """
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])

        # Pooling для уменьшения разрешения
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, x):
        """
        x: (B, 1, T)
        return: list of (output, feature_maps) для каждого дискриминатора
        """
        outputs = []
        for i, disc in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i - 1](x)
            outputs.append(disc(x))
        return outputs
