import numpy as np
from nltk.stem.porter import PorterStemmer
from collections import Counter

stemmer = PorterStemmer()


class Loader:
    def __init__(self):
        self.train_path = './data/training/'
        self.dev_x_path = './scripts/similarity/dev_x.csv'
        self.dev_y_path = './scripts/similarity/dev_y.csv'
        self.test_x_path = './scripts/similarity/test_x.csv'

    def load_data(self, dataset):
        text_word = []
        text_id = []
        vocab = {}
        rev_vocab = {}
        stems = {}
        # go through the whole data
        with open(self.train_path + dataset, 'r') as f:
            for line in f:
                line = line.split()
                for word in line:
                    text_word.append(word)
        # build vocabulary starting from the lowest index/most common word
        for i, pair in enumerate(Counter(text_word).most_common()):
            word, _ = pair
            vocab[word] = i
            rev_vocab[i] = word
            stems[stemmer.stem(word)] = word # stemmed word to word mapping
        for word in text_word:
            text_id.append(vocab[word])
        counts = dict(Counter(text_word).most_common())
        return text_id, vocab, rev_vocab, counts, stems

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
    text, vocab, rev_vocab, counts, stems = loader.load_data('data1m')
    print('Data size:', len(text))
    print('Vocabulary size:', len(vocab))

