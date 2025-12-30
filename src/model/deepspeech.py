from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DeepSpeech2(nn.Module):
    def __init__(self, n_feats, n_tokens, rnn_hidden=512, num_rnn_layers=3):
        super().__init__()

        # Сверточный блок:
        # - уменьшает время в 2 раза на каждом слое с stride=(2, 1)
        # - уменьшает частоты в 2 раза на каждом слое с stride=(2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        )

        # Conv1: H_out = floor((n_feats + 2*20 - 41)/2 + 1) = floor(n_feats/2)
        # Conv2: H_out = floor((H_out + 2*10 - 21)/2 + 1) = floor(H_out/2)

        rnn_input_size = n_feats
        rnn_input_size = (rnn_input_size + 2 * 20 - 41) // 2 + 1
        rnn_input_size = (rnn_input_size + 2 * 10 - 21) // 2 + 1
        rnn_input_size *= 32  # channels

        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.fc = nn.Linear(rnn_hidden * 2, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        # spectrogram: (B, F, T) -> (B, 1, F, T)
        x = spectrogram.unsqueeze(1)

        # Conv
        x = self.conv(x)

        # (B, C, F_new, T_new) -> (B, T_new, C * F_new)
        B, C, F_new, T_new = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T_new, -1)

        # Recalculate lengths
        new_lengths = self.transform_input_lengths(spectrogram_length)

        # Pack padded sequence
        x = pack_padded_sequence(
            x, new_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        output = self.fc(x)
        log_probs = nn.functional.log_softmax(output, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": new_lengths}

    def transform_input_lengths(self, input_lengths):
        # L_out = (L_in + 2*padding - kernel) // stride + 1

        # Conv 1
        # kernel=(41, 11), stride=(2, 2), padding=(20, 5)
        new_lengths = (input_lengths + 2 * 5 - 11) // 2 + 1

        # Conv 2
        # kernel=(21, 11), stride=(2, 1), padding=(10, 5)
        new_lengths = (new_lengths + 2 * 5 - 11) // 1 + 1

        return new_lengths
