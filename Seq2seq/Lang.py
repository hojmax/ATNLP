import pickle


class Lang:
    # Static variables
    SOS_token = 0
    EOS_token = 1
    PAD_token = 2

    def __init__(self):
        """Initiates a language object containing a vocabulary of words and their
           corresponding indices.

        Args:
            name (str): A name for the language
        """
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            self.SOS_token: "SOS",
            self.EOS_token: "EOS",
            self.PAD_token: "PAD"
        }
        self.n_words = 3  # Count SOS, EOS and PAD

    def add_sentence(self, sentence):
        """Adds a sencente to the vocabulary by splitting by spaces and adding each word

        Args:
            sentence (str): the sentence to add
        """
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """Adds a word to the vocabulary

        Args:
            word (str): the word to add
        """
        if word not in self.word2index:           # If the word is not in the vocabulary
            self.word2index[word] = self.n_words  # Add it to the vocabulary
            self.word2count[word] = 1             # Set the count to 1
            self.index2word[self.n_words] = word  # Add the word to the index
            self.n_words += 1                     # Increment the number of words
        else:
            # Increment the count of the word
            self.word2count[word] += 1

    def save(self, filename):
        """Saves the language object to a file

        Args:
            filename (str): The filename to save to
        """
        with open(filename, 'wb') as f:
            pickle.dump(
                (
                    self.word2index,
                    self.word2count,
                    self.index2word,
                    self.n_words
                ),
                f
            )

    def load(self, filename):
        """Loads a language object from a file

        Args:
            filename (str): The filename to load from
        """
        with open(filename, 'rb') as f:
            (
                self.word2index,
                self.word2count,
                self.index2word,
                self.n_words
            ) = pickle.load(f)
