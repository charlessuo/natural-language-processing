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
        idx = 1
        with open('./data/training/' + dataset, 'r') as f:
            for line in f:
                line = line.split()
                for word in line:
                    #if not word.isalpha():
                    #    continue
                    #if word in string.punctuation:
                    #    continue            
                    #word = lemmatizer.lemmatize(word).lower()
                    #word = word.lower()
                    if word not in vocab:
                        vocab[word] = idx
                        rev_vocab[idx] = word
                        text.append(idx)
                        counts[word] = 1
                        idx += 1
                    else:
                        text.append(vocab[word])
                        counts[word] += 1

        for i in range(len(text)):
            word = rev_vocab[text[i]]
            if counts[word] < 2:
                text[i] = 0 # set to UNK
                vocab[word] = 0
                rev_vocab[0] = 'UKN'
        return text, vocab, rev_vocab

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
#    texts, vocab, rev_vocab = loader.load_data()
    word_pairs, sim_lables = loader.load_eval()
    word_pairs = loader.load_test()
