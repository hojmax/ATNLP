import torch
from Lang import Lang
import random


def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    input_max_length,
    teacher_forcing_ratio,
    gradient_clip_norm
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder.train()
    decoder.train()
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        input_max_length, encoder.hidden_size, device=device
    )

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei],
            encoder_hidden
        )
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor(
        [[Lang.SOS_token]],
        device=device
    )

    decoder_hidden = encoder_hidden

    use_teacher_forcing = random.random() < teacher_forcing_ratio

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs,# Only for attention
            )
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs, #Only for attention
            )
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == Lang.EOS_token:
                break

    loss.backward()

    torch.nn.utils.clip_grad_value_(encoder.parameters(), gradient_clip_norm)
    torch.nn.utils.clip_grad_value_(decoder.parameters(), gradient_clip_norm)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
