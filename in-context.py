from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline, set_seed
from datasets import load_dataset
from tqdm import tqdm
import argparse
import colored
import torch
import wandb
import json
import os


def print_green(text):
    print(colored.fg('green') + text + colored.attr('reset'))


if __name__ == '__main__':
    wandb_project = "individual-atnlp"
    wandb_entity = "hojmax"

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str,
                           default="facebook/opt-6.7b")
    argparser.add_argument('--dataset', type=str,
                           default='top16.json')
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--wandb-key', type=str)
    argparser.add_argument('--max-examples', type=int)
    config = vars(argparser.parse_args())

    wandb.login(
        # Remove key from config
        key=config.pop('wandb_key')
    )

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config=config,
    )

    print_green('Loading dataset...')
    dataset = load_dataset(
        "json",
        data_files=f'prompts/{config["dataset"]}'
    )['train']

    print_green('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        config['model'],
        torch_dtype=torch.float16
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        config['model'],
        use_fast=False
    )

    if config['max_examples']:
        dataset = dataset.select(range(config['max_examples']))

    print_green('Generating...')
    outputs = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in tqdm(range(len(dataset))):
        input_ids = tokenizer(
            dataset['prompt'][i],
            return_tensors="pt"
        ).input_ids.cuda()
        # Max target tokens is 81, so 100 is generous
        generated_ids = model.generate(input_ids, max_new_tokens=100)
        outputs.append(
            tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0][len(dataset['prompt'][i]):].split('\n')[0].strip()
        )

    if config['debug']:
        for i, output in enumerate(outputs):
            print(colored.fg('red') + 'Prompt:' + colored.attr('reset'))
            print(dataset['prompt'][i])
            print(colored.fg('red') + 'Output: ' +
                  colored.attr('reset') + output)
            print(colored.fg('red') + 'Target: ' +
                  colored.attr('reset') + dataset['target'][i])
            print()

    print_green('Evaluating...')
    correct = 0
    for i, output in enumerate(outputs):
        if output == dataset['target'][i]:
            correct += 1

    run.log({
        'accuracy': correct / len(outputs)
    })

    print(
        colored.fg('green') +
        'Accuracy: ' +
        colored.attr('reset') +
        str(correct / len(outputs) * 100) + '%'
    )

    with open('outputs.json', 'w') as f:
        json.dump(outputs, f)

    wandb.save('outputs.json')

    run.finish()
