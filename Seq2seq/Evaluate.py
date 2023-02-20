from Seq2seq.Lang import Lang
import numpy as np
from scipy.spatial import distance
import pandas as pd
import torch
from matplotlib import pyplot as plt


def get_accuracy_across_length(filter, test_X, test_Y, encoder, decoder, input_max_length, oracle: bool = False):
    # -1 because of the EOS token
    all_lengths = np.array([len(x)-1 for x in filter])
    unique_lengths = np.unique(all_lengths)
    results = []
    for length in unique_lengths:
        mask = all_lengths == length
        results.append([
            length,
            get_accuracy(
                [x for i, x in enumerate(test_X) if mask[i]],
                [y for i, y in enumerate(test_Y) if mask[i]],
                encoder,
                decoder,
                input_max_length,
                oracle  # additional argument for oracle case
            )
        ])
    return results


def get_accuracy(test_X, test_Y, encoder, decoder, input_max_length, oracle: bool = False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder.eval()
    decoder.eval()
    n_matches = 0

    with torch.no_grad():
        for input_tensor, target_tensor in zip(test_X, test_Y):
            encoder_hidden = encoder.initHidden()

            input_length = input_tensor.size(0)

            encoder_outputs = torch.zeros(
                input_max_length, encoder.hidden_size, device=device
            )

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

            decoded_words = []

            while True:
                decoder_output, decoder_hidden = decoder(
                    decoder_input,
                    decoder_hidden,
                    encoder_outputs,  # Only for attention
                )
                topv, topi = decoder_output.topk(1)
                decoded_words.append(topi.item())
                decoder_input = topi.squeeze().detach()

                # revised for oracle case
                if len(decoded_words) > len(target_tensor):
                    break

                if decoder_input.item() == Lang.EOS_token:
                    if not oracle:
                        break
                    elif oracle and len(decoded_words) < len(target_tensor):
                        # take the second candidate
                        decoded_words.pop()
                        top_2nd_v, top_2nd_i = decoder_output.topk(2)
                        decoded_words.append(top_2nd_i.squeeze().tolist()[1])
                        decoder_input = top_2nd_i.squeeze()[1].detach()
                    else:
                        break

            decoded_words = torch.tensor(
                decoded_words,
                device=device
            ).reshape(-1, 1)

            if torch.equal(decoded_words, target_tensor):
                n_matches += 1

    return n_matches / len(test_X)


def get_cosine_word_df():
    words_df = pd.DataFrame({
        "run": [
            "look",
            "walk",
            "walk after run",
            "run thrice after run",
            "run twice after run"
        ],
        "jump": [
            "run",
            "walk",
            "turn right",
            "look right twice after walk twice",
            "turn right after turn right"
        ],
        "run twice": [
            "look twice",
            "run twice and look opposite right thrice",
            "run twice and run right twice",
            "run twice and look opposite right twice",
            "walk twice and run twice"
        ],
        "jump twice": [
            "walk and walk",
            "run and walk",
            "walk opposite right and walk",
            "look right and walk",
            "walk right and walk"
        ]
    })

    return words_df


def get_cosine_table2(encoder, dataloader):
    encoder.eval()
    all_X, _ = dataloader.transform(
        'SCAN/add_prim_split/tasks_train_addprim_jump.txt')
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    df = get_cosine_word_df()

    all_encoding_names = []
    all_encodings = []

    print("Encoding all words...")
    headers = []
    word_matrix = []
    value_matrix = []

    for i, x in enumerate(all_X):
        v2 = encoder.encode(x)[1][0][-1].squeeze(0).detach()
        all_encodings.append(v2)
        all_encoding_names.append(dataloader.pairs[i][0])

    for i, n1 in enumerate(df.columns):
        i1 = dataloader.transform_string(n1)
        v1 = encoder.encode(i1)[1][0][-1].squeeze(0).detach()

        row = []
        names_ = []
        print(f'Word: {n1}')
        headers.append(n1)

        for i, encoding in enumerate(all_encodings):
            if all_encoding_names[i] == n1:
                continue
            row.append(cos(v1, encoding).item())
            names_.append(all_encoding_names[i])

        # Print top 6
        top_5 = np.argsort(row)[::-1][:5]
        word_matrix.append([])
        value_matrix.append([])

        for i in top_5:
            print(f"{names_[i]}: {row[i]}")
            word_matrix[-1].append(names_[i])
            value_matrix[-1].append(row[i])

        print()

    # Create pandas dataframe
    # With cells like: f'{word}: {value}'
    # This combines word_matrix and value_matrix
    return headers, word_matrix, value_matrix


def get_cosine_table(encoder, dataloader):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    encoder.eval()
    res = []
    df = get_cosine_word_df()

    for i, n1 in enumerate(df.columns):
        i1 = dataloader.transform_string(n1)
        v1 = encoder.encode(i1)[1][0][-1].squeeze(0).detach()

        for n2 in df[n1].values:
            i2 = dataloader.transform_string(n2)
            v2 = encoder.encode(i2)[1][0][-1].squeeze(0).detach()
            sim = cos(v1, v2).item()
            print(f"{n1} vs {n2}: {sim:.2f}")
            res.append(sim)

    results = np.array(res).reshape(4, 5).T
    results_df = pd.DataFrame(data=results, columns=df.columns)

    return df, results_df


def get_cosine_latex_df(df, results_df):
    latex_df = results_df.copy()

    def mlticol(x):
        return " & \\multicolumn{1}{r}{" + x + "}"

    for n1, c in enumerate(results_df.columns):
        for n2 in range(0, len(results_df[c])):
            v = results_df.iloc[n2, n1]
            vformat = "{:.2f}".format(v)
            vformat = "\\textit{" + vformat + "}" if v < 0.2 else vformat
            latex_df.iloc[n2, n1] = df.iloc[n2, n1] + mlticol(vformat)

    latex_df.columns = [
        "\\multicolumn{2}{c}{" + c + "}" for c in latex_df.columns
    ]

    return latex_df


def generate_cosine_latex_table(encoder, dataloader):
    df, results_df = get_cosine_table(encoder, dataloader)
    latex_df = get_cosine_latex_df(df, results_df)
    with pd.option_context("max_colwidth", 1000):
        latex_table = latex_df.to_latex(index=False)

    latex_table = latex_table.replace("llll", "llllllll")
    latex_table = latex_table.replace("\\&", "&")
    latex_table = latex_table.replace("\\}", "}")
    latex_table = latex_table.replace("\\{", "{")
    latex_table = latex_table.replace("\\textbackslash ", "\\")

    return latex_table


def plot_attention(encoder,
                   decoder,
                   dataloader,
                   input_string,
                   figsize=(14, 6),
                   filename=None):
    # Set both functions to eval
    encoder.eval()
    decoder.eval()

    # encode input string
    vect_string = dataloader.transform_string(input_string)

    # get encoder outputs and hidden state from encoder
    encoder_outputs, encoder_hidden_state = encoder.encode(vect_string)

    # get decoder outputs and attention weights from decoder
    decoded_words, attns = decoder.decode(
        encoder_outputs,
        encoder_hidden_state,
        return_attention=True
    )

    # convert decoded_words into string
    decoded_result = dataloader.decode_string(decoded_words)

    # Squeeze attention into 2D format (seq_len_in, seq_len_out)
    attns = attns.squeeze(1).transpose(0, 1)

    # Plot attention as heatmap using plt
    # Fill each heatmap field with value with white text
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(attns.numpy())
    for (i, j), z in np.ndenumerate(attns):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center',
                va='center', color="white")

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(attns.numpy())
    fig.colorbar(cax)

    # in decoded_result, remove "EOS"
    decoded_result = decoded_result.split(" ")
    decoded_result.remove("EOS")
    decoded_result = " ".join(decoded_result)

    # make one y tick for each word in input_string
    ax.set_yticks(range(len(input_string.split(" ")) + 1))
    # make one x tick for each word in decoded_result
    ax.set_xticks(range(len(decoded_result.split(" ")) + 1))

    # make each word of decoded_result a tick on the x axis
    ax.set_xticklabels(["<SOS>"] + decoded_result.split(" "), rotation=45)

    # make each word of input_string a tick on the y axis
    ax.set_yticklabels(input_string.split(" ") + ["<EOS>"])

    if filename is not None:
        plt.savefig(filename,
                    bbox_inches="tight",
                    dpi=300)

    plt.show()
