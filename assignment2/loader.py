import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


class Loader:
    def __init__(self):
        self.train_path = './data/training/'
        self.dev_x_path = './scripts/similarity/dev_x.csv'
        self.dev_y_path = './scripts/similarity/dev_y.csv'
        self.test_x_path = './scripts/similarity/test_x.csv'

    def load_data(self, dataset):
        text = []
        vocab = {}
        rev_vocab = {}
        counts = {}
        stems = {}
        idx = 0
        with open(self.train_path + dataset, 'r') as f:
            for line in f:
                line = line.split()
                for word in line:
                    if word not in vocab:
                        vocab[word] = idx
                        rev_vocab[idx] = word
                        text.append(idx)
                        counts[word] = 1
                        idx += 1
                        stems[stemmer.stem(word)] = word
                    else:
                        text.append(vocab[word])
                        counts[word] += 1
        return text, vocab, rev_vocab, counts, stems

    def load_eval(self):
        word_pairs = []
        with open(self.dev_x_path) as fx:
            next(fx)
            for line in fx:
                word_pairs.append(line.strip().split(',')[1:])

        similarities = []
        with open(self.dev_y_path) as fy:
            next(fy)
            for line in fy:
                similarities.append(float(line.strip().split(',')[1]))
        return word_pairs, similarities

    def load_test(self):
        word_pairs = []
        with open(self.test_x_path) as fx:
            next(fx)
            for line in fx:
                word_pairs.append(line.strip().split(',')[1:])
        return word_pairs


if __name__ == '__main__':
    loader = Loader()
    text, vocab, rev_vocab, counts, stems = loader.load_data('data1to30_9M')
    print('Data size:', len(text))
    print('Vocabulary size:', len(vocab))
#    word_pairs, sim_lables = loader.load_eval()
#    word_pairs = loader.load_test()

