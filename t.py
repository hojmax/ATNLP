from Dataloader import Dataloader
from Seq2SeqTransformer import Seq2SeqTransformer, create_mask, generate_square_subsequent_mask
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
from Lang import Lang
import torch
import wandb
import os
os.environ['WANDB_NOTEBOOK_NAME'] = 'transformer.ipynb'
wandb.login()

config = {
    'learning_rate': 0.001,
    'dropout': 0.1,
    'hidden_size': 200,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'nhead': 4,
    'epochs': 20,
    'train_path': 'SCAN/simple_split/tasks_train_simple.txt',
    'test_path': 'SCAN/simple_split/tasks_test_simple.txt',
    'dataset': 'simple',
    'batch_size': 1,
}

dataloader = Dataloader()

train_X, train_Y = dataloader.fit_transform(config['train_path'])
test_X, test_Y = dataloader.transform(config['test_path'])

config['input_size'] = dataloader.input_lang.n_words
config['output_size'] = dataloader.output_lang.n_words

wandb.init(
    project="individual-atnlp",
    entity="hojmax",
    name=f"Transformer, Dataset: {config['dataset']}",
    config=config,
    tags=["test"]
)
dataloader.save(wandb.run.dir)

transformer = Seq2SeqTransformer(
    num_encoder_layers=config['num_encoder_layers'],
    num_decoder_layers=config['num_decoder_layers'],
    emb_size=config['hidden_size'],
    nhead=config['nhead'],
    src_vocab_size=config['input_size'],
    tgt_vocab_size=config['output_size'],
    dim_feedforward=config['hidden_size'],
    dropout=config['dropout']
)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=Lang.PAD_token)
optimizer = torch.optim.Adam(
    transformer.parameters(),
    lr=config['learning_rate']
)

train_X = [e.flatten() for e in train_X]
train_Y = [e.flatten() for e in train_Y]

train_inputs = pad_sequence(
    train_X,
    padding_value=Lang.PAD_token,
    batch_first=True
)
train_targets = pad_sequence(
    train_Y,
    padding_value=Lang.PAD_token,
    batch_first=True
)

train_dataset = TensorDataset(train_inputs, train_targets)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def train_epoch(model, optimizer):
    model.train()
    losses = 0

    for src, tgt in train_dataloader:
        # .T because dataloader is returning batch_first
        src = src.T.to(device)
        tgt = tgt.T.to(device)
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src,
            tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            tgt_out.reshape(-1)
        )
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


for epoch in range(config['epochs']):
    loss = train_epoch(transformer, optimizer)
    wandb.log({
        "avg_epoch_loss": loss,
        "epoch": epoch + 1
    })


torch.save(transformer.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
