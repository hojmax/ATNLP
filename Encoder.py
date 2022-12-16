import torch.nn as nn
import torch


class EncoderRNN(nn.Module):
    def __init__(
        self,
        RNN_type,
        input_size,
        hidden_size,
        hidden_layers,
        dropout
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.dropout_prob = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.RNN_type = RNN_type
        # RNN_type can be 'LSTM', 'GRU' or 'RNN'
        self.rnn = nn.__dict__[RNN_type](
            input_size=self.hidden_size,  # input_size is hidden_size since we are using embedding
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layers,
            dropout=self.dropout_prob
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, input, hidden):
        # NOTE: forward takes only one example at a time and thus we need to
        # reshape the tensor to be able to use it with rnns
        output = self.embedding(input).view(1, 1, -1)
        output = self.dropout(output)
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        if self.RNN_type == 'LSTM':
            return (
                torch.zeros(
                    self.hidden_layers,
                    1,
                    self.hidden_size,
                    device=self.device
                ),
                torch.zeros(
                    self.hidden_layers,
                    1,
                    self.hidden_size,
                    device=self.device
                )
            )
        else:
            return torch.zeros(
                self.hidden_layers,
                1,
                self.hidden_size,
                device=self.device
            )

    def encode(self, input_tensor):
        encoder_hidden = self.initHidden()

        input_length = input_tensor.size(0)

        encoder_outputs = torch.zeros(
            input_length, self.hidden_size, device=self.device
        )

        for ei in range(input_length):
            encoder_output, encoder_hidden = self(
                input_tensor[ei],
                encoder_hidden
            )
            encoder_outputs[ei] = encoder_output[0, 0]

        return encoder_outputs, encoder_hidden
