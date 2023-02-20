import torch.nn as nn
import torch
from Seq2seq.attention import Attention
from Seq2seq.Lang import Lang


class DecoderRNN(nn.Module):
    def __init__(
        self,
        RNN_type,
        input_size,
        output_size,
        hidden_size,
        hidden_layers,
        dropout,
        attention=False,  # For future use
        linear_out_fix=False,
        sum_attention_flag=False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.input_size = input_size
        self.dropout_prob = dropout

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        # RNN_type can be 'LSTM', 'GRU' or 'RNN'
        self.model = nn.__dict__[RNN_type](
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layers,
            dropout=self.dropout_prob
        )

        # Internal model parameters
        self.relu = nn.ReLU()
        self.linear_out_fix = linear_out_fix
        self.sum_attention_flag = sum_attention_flag
        if linear_out_fix:
            self.linear = nn.Linear(self.hidden_size, self.output_size)
        else:
            self.linear_out = nn.Linear(self.hidden_size, self.output_size)

        self.softmax = nn.LogSoftmax(dim=1)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Attention
        self.has_attention = attention
        if self.has_attention:
            self.attention = Attention(self.hidden_size)
            self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.attn_concat = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self,
                input,
                prev_decoder_hidden,
                encoder_outputs,
                return_attention=False):
        # input is a single word, not a batch
        # After embedding a word we thus change the shape from (hidden_size)
        # to (1, 1, hidden_size)
        embedding = self.embedding(input).view(1, 1, -1)
        embedding = self.dropout(embedding)
        embedding = self.relu(embedding)

        if self.has_attention:
            # If attention is turned on we compute the context vector,
            # concatenate it with the original embedding and map it to to have
            # dimensionality equal to the hidden size
            context = self.attention(
                prev_decoder_hidden,
                encoder_outputs,
                return_attention=return_attention
            )

            if return_attention:
                context, attn = context

            if self.sum_attention_flag:
                embedding = embedding + context.unsqueeze(1)
            else:
                embedding = torch.cat((embedding, context.unsqueeze(1)), dim=-1)
                embedding = self.attn_combine(embedding)

        # embedding = self.relu(embedding)

        # We pass the embedding through the RNN one time step,
        # computing the new hidden state and the output
        output, decoder_hidden = self.model(embedding, prev_decoder_hidden)

        # output = self.relu(output)

        if self.has_attention:
            attn_cat = torch.cat((output, context.unsqueeze(1)), dim=-1)
            output = self.attn_concat(attn_cat)
            # output = self.relu(output)

        # We map the output to the vocabulary size and do a softmax
        output = self.linear_out(output[0])
        output = self.softmax(output)

        if return_attention and self.has_attention:
            return output, decoder_hidden, attn
        return output, decoder_hidden

    def initHidden(self):
        return torch.zeros(
            self.hidden_layers,
            1,
            self.hidden_size,
            device=self.device
        )

    def decode(self,
               encoder_outputs,
               decoder_hidden,
               return_attention=False):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        decoder_input = torch.tensor(
            [[Lang.SOS_token]],
            device=device
        )

        decoded_words = []

        attns = []

        while True:
            if return_attention:
                decoder_output, decoder_hidden, attn = self(
                    decoder_input,
                    decoder_hidden,
                    encoder_outputs,  # Only for attention
                    return_attention=return_attention
                )

                attns.append(attn.detach().cpu())
            else:
                decoder_output, decoder_hidden = self(
                    decoder_input,
                    decoder_hidden,
                    encoder_outputs  # Only for attention
                )
            topv, topi = decoder_output.topk(1)
            decoded_words.append(topi.item())
            decoder_input = topi.squeeze().detach()

            if decoder_input.item() == Lang.EOS_token or len(decoded_words) > 64:
                break

        decoded_words = torch.tensor(
            decoded_words,
            device=device
        ).reshape(-1, 1)

        attns = torch.stack(attns)

        if return_attention:
            return decoded_words, attns
        return decoded_words
