from torch import nn


class BaselineModel(nn.Module):
    """
    Simple MLP + RNN Baseline
    """

    def __init__(self, n_feats, n_tokens, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features (frequency dimension).
            n_tokens (int): number of tokens in the vocabulary (num classes).
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        # Немного другой baseline
        self.rnn = nn.GRU(
            input_size=n_feats,
            hidden_size=fc_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(in_features=fc_hidden * 2, out_features=n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Args:
            spectrogram (Tensor): input spectrogram (Batch, Freq, Time).
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and transformed lengths.
        """
        # RNN ожидает (Batch, Time, Features)
        # Вход у нас (Batch, Freq, Time)
        x = spectrogram.transpose(1, 2)

        output, _ = self.rnn(x)
        output = self.fc(output)

        # Log Softmax для CTC Loss
        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths  # we don't reduce time dimension here

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
