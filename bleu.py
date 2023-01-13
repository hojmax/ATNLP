from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def extract_input(line):
    return line.split(' OUT: ')[0][len('IN: '):]


raw_lines = open(
    "./SCAN/add_prim_split/tasks_train_addprim_jump.txt", 'r').readlines()
raw_inputs = [extract_input(line) for line in raw_lines]
raw_inputs_splitted = [line.split() for line in raw_inputs]


def get_top_closest(query, k=5):
    smoothing = SmoothingFunction().method4
    scores = [
        sentence_bleu([candidate], query.split(), smoothing_function=smoothing) for candidate in raw_inputs_splitted
    ]
    return [raw_inputs[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]]


query = 'turn left after jump twice'
k = 30
candidates = get_top_closest(query, k)
print(f'Query: {query}')
print(f'Top {k} candidates:')
for candidate in candidates:
    print(candidate)
print(sentence_bleu([['jump']], query.split(),
      smoothing_function=SmoothingFunction().method4))
