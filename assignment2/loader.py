import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
import string

lemmatizer = WordNetLemmatizer()


class Loader:
    def __init__(self):
        pass

    def load_data(self, dataset):
        text = []
        vocab = {}
        rev_vocab = {}
        counts = {}
        idx = 0
        with open('./data/training/' + dataset, 'r') as f:
            for line in f:
                line = line.split()
                for word in line:
                    if word not in vocab:
                        vocab[word] = idx
                        rev_vocab[idx] = word
                        text.append(idx)
                        counts[word] = 1
                        idx += 1
                    else:
                        text.append(vocab[word])
                        counts[word] += 1
        return text, vocab, rev_vocab, counts

    def load_eval(self):
        word_pairs = []
        with open('./scripts/similarity/dev_x.csv') as fx:
            next(fx)
            for line in fx:
                word_pairs.append(line.strip().split(',')[1:])

        similarities = []
        with open('./scripts/similarity/dev_y.csv') as fy:
            next(fy)
            for line in fy:
                similarities.append(float(line.strip().split(',')[1]))
        return word_pairs, similarities

    def load_test(self):
        word_pairs = []
        with open('./scripts/similarity/test_x.csv') as fx:
            next(fx)
            for line in fx:
                word_pairs.append(line.strip().split(',')[1:])
        return word_pairs


if __name__ == '__main__':
    loader = Loader()
    text, vocab, rev_vocab, counts = loader.load_data('data1to20_6M')
    print('Data size:', len(text))
    print('Vocabulary size:', len(vocab))
#    word_pairs, sim_lables = loader.load_eval()
#    word_pairs = loader.load_test()

