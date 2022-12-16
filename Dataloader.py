from Lang import Lang
import torch
import os


def get_dataset_path(folder, dataset, type):
    return f'{folder}/tasks_{type}_{dataset}.txt'


class Dataloader:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # -1 since max_length is not known yet
        self.input_max_length = -1

    def fit_transform(self, path):
        """Basic fit and transform function for the dataloader
           gien a path to a dataset (a file of the appropriate format)
           this generates and returns a tuple of lists of tensors -
           one for X and one for y. This also trains the dataloader on
           how to transform the dataset.

        Args:
            path (str): the path to the dataset

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                A tuple of lists of tensors. Functionas as the dataset
        """
        # Converts the sencences from the dataset into pairs
        # pairs are a list of tuples of strings (input, output)
        self.pairs = self._get_pairs(path)
        # Get the input and output languages
        self.input_lang, self.output_lang = self._get_langs(self.pairs)
        # Get the tensors from the pairs
        x, y = self._get_tensors(
            self.input_lang,
            self.output_lang,
            self.pairs
        )
        self._update_input_max_length(x)
        return x, y

    def transform(self, path):
        """Transforms a dataset into a list of tensors

        Args:
            path (str): The path to the dataset

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                A tuple of lists of tensors. Functionas as the dataset

        """
        self.pairs = self._get_pairs(path)
        x, y = self._get_tensors(
            self.input_lang,
            self.output_lang,
            self.pairs
        )
        self._update_input_max_length(x)
        return x, y

    def transform_string(self, string):
        x, _ = self._get_tensors(
            self.input_lang,
            self.output_lang,
            [[string, "I_JUMP"]]
        )
        return x[0]

    def decode_string(self, outs):
        return " ".join([
            self.output_lang.index2word[i.item()] for i in outs
        ])

    def save(self, path):
        """A basic saving function for the dataloader"""
        self.input_lang.save(os.path.join(path, 'input_lang.pkl'))
        self.output_lang.save(os.path.join(path, 'output_lang.pkl'))

        with open(os.path.join(path, 'max_length.txt'), 'w') as f:
            f.write(str(self.input_max_length))

        return self

    def load(self, path):
        """A basic loading function for the dataloader"""
        self.input_lang = Lang("input")
        self.output_lang = Lang("output")
        self.input_lang.load(os.path.join(path, 'input_lang.pkl'))
        self.output_lang.load(os.path.join(path, 'output_lang.pkl'))

        with open(os.path.join(path, 'max_length.txt'), 'r') as f:
            self.input_max_length = int(f.read())

        return self

    def _get_pairs(self, path):
        """Given a path, creates a list of pairs of strings (input, output)

        Args:
            path (str): The path to the dataset

        Returns:
            Tuple[List[str], List[str]]: The list of pairs
        """
        pairs = []
        seperator_token = ' OUT: '
        trailing_token = 'IN: '
        lines = open(path, 'r').readlines()
        for line in lines:
            pairs.append(
                line.strip()
                    .replace(trailing_token, '')
                    .split(seperator_token)
            )
        return pairs

    def _update_input_max_length(self, x):
        """Updates the internal maximum length state

        Args:
            x (int): The new maximum length
        """
        current_max_length = max([len(i) for i in x])
        self.input_max_length = max(
            self.input_max_length,
            current_max_length
        )

    def _get_langs(self, pairs):
        """_summary_

        Args:
            pairs (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_lang = Lang("input")
        output_lang = Lang("output")
        for pair in pairs:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])
        return input_lang, output_lang

    def _index_from_sentence(self, lang, sentence):
        """Converts a centence to a list of indices using the given language.
           The sentence is split by spaces (" ").

        Args:
            lang (Lang): A language object
            sentence (str): A sentence

        Returns:
            List[int]: A list of indices
        """
        return [lang.word2index[word] for word in sentence.split()]

    def _tensor_from_sentence(self, lang, sentence):
        """creates a tensor of indices from a sentence split by spaces given a language

        Args:
            lang (Lang): A language
            sentence (str): A sentence

        Returns:
            torch.Tensor: A tensor of shape (len(sentence), 1)
        """
        indexes = self._index_from_sentence(lang, sentence)
        indexes.append(lang.EOS_token)
        return torch.tensor(
            indexes,
            dtype=torch.long,
            device=self.device
        ).view(-1, 1)

    def _get_tensors(self, input_lang, output_lang, pairs):
        """Given an input and output language and a list of pairs of strings,
           returns a list of tensors for the input and output

        Args:
            input_lang (Lang): An input language
            output_lang (Lang): an output language
            pairs (Tuple[List[str], List[str]]): The pairs representing the dataset

        Returns:
            Tuple[List[str], List[str]]: A tuple of lists of tensors representing the dataset
        """
        X = []
        Y = []

        for pair in pairs:
            X.append(self._tensor_from_sentence(input_lang, pair[0]))
            Y.append(self._tensor_from_sentence(output_lang, pair[1]))

        return X, Y
