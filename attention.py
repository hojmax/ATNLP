import torch.nn.functional as F
import torch.nn as nn
import torch


class Attention(nn.Module):
    """Attention module as described in the paper supplementary material.
       Computes additive attention across all encoder outputs for each given
       hidden input state of the decoder.

       Bahdanau attention returns a context vector that is meant to be handled
       in conjunction with the decoder hidden state after its computation.
    """

    def __init__(self, hidden_size):
        """Initializes the attention module.
        Args:
            hidden_size (int): The hidden size of the model.
            Must be the same as the hidden size of the encoder and decoder
        """
        super().__init__()
        self.uA = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wA = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vA = nn.Linear(hidden_size, 1, bias=False)

    def attention(self, s, h):
        """Computes the additive attention score for each encoder output.

        Args:
            s (torch.Tensor): The previous decoder hidden state. If LSTM, this must be
                only the hidden state and not the cell state. If num_layers > 1, this
                must be the hidden state of the last layer.
            h (torch.Tensor): The encoder outputs.

        Returns:
            torch.Tensor: the result of additive attention
        """
        # Attention mechanism as described in the paper supplementary material
        u_ = self.uA(s)
        w_ = self.wA(h)
        # Broadcasted addition on u_ and w_ with w_ as base
        # Is used as we calculate all w_ simultaneously in each fwd pass
        tmp = torch.tanh(u_ + w_)  # Produces (n, hidden_size) where n is the hidden dimension
        v = self.vA(tmp)  # Produces (n, 1)
        return v.permute(1, 0)  # Produces (1, n)

    def forward(self,
                prev_decoder_hidden,
                encoder_outputs,
                return_attention=False):
        """Computes the Bahdanau attention score for each encoder output.

        Args:
            prev_decoder_hidden (torch.Tensor): The previous decoder hidden state.
            encoder_outputs (torch.Tensor): The encoder outputs.

        Returns:
            torch.Tensor: the result of Bahdanau attention
        """
        # Get the last layer of the last decoder hidden state
        last_decoder_hidden_layer = prev_decoder_hidden[0][-1]

        es = self.attention(last_decoder_hidden_layer, encoder_outputs)

        alphas = F.softmax(es, dim=1)

        context = alphas @ encoder_outputs  # Produces a (1, n) tensor

        if return_attention:
            return context, alphas

        return context

    def attend(self, prev_decoder_hidden, encoder_outputs):
        last_decoder_hidden_layer = prev_decoder_hidden[0][-1]

        es = self.attention(last_decoder_hidden_layer, encoder_outputs)

        alphas = F.softmax(es, dim=1)

        return alphas
