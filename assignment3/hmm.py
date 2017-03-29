#!/usr/bin/env python3
import numpy as np
from hmm_loader import Loader
from collections import defaultdict

class HMM:
    def __init__(self):
        self.emissions = None
        self.transitions = None
        self.states = {}

    def train(self, train_x, train_y, ngram, smooth):
        self.emissions = self._emisson_probs(train_x, train_y, smooth)
        self.transitions = self._transition_probs(train_y, ngram, smooth)

    def _transition_probs(self, train_y, ngram, smooth):
        if smooth == 'add_one':
            return self._transition_add_one(train_y, ngram)
        elif smooth == 'linear_interpolate':
            return self._transition_linear_interpolate(train_y, ngram)

    def _emisson_probs(self, train_x, train_y, smooth):
        if smooth == 'add_one':
            return self._emisson_add_one(train_x, train_y)
        elif smooth == 'linear_interpolate':
            return self._emission_linear_interpolate(train_x, train_y)

    def _emission_linear_interpolate(self, train_x, train_y):
        pass

    def _transition_linear_interpolate(self, train_y, ngram):
        pass

    def _transition_add_one(self, train_y, ngram):
        transitions = {}
        tag_set = set()
        nomin_count = defaultdict(int)
        denom_count = defaultdict(int)
        for tags in train_y:
            for i, tag in enumerate(tags):
                if tag is None:
                    continue
                tag_set.add(tag)
                if ngram == 2:
                    N, V = tags[i], tags[i - 1]
                    nomin_count[(N, V)] += 1
                    denom_count[V] += 1
                elif ngram == 3:
                    N, V, D = tags[i], tags[i - 1], tags[i - 2]
                    nomin_count[(N, V, D)] += 1
                    denom_count[(V, D)] += 1
        tag_list = list(tag_set)

        if ngram == 2:
            tag_pairs = [(tag1, tag2) for tag1 in tag_list for tag2 in tag_list]
            for tag_pair in tag_pairs:
                N, V = tag_pair
                if (N, V) in nomin_count:
                    transitions[(N, V)] = np.log((nomin_count[(N, V)] + 1) / (denom_count[V] + len(denom_count)))
                else:
                    transitions[(N, V)] = np.log(1 / (denom_count[V] + len(denom_count)))
        if ngram == 3:
            tag_pairs = [(tag1, tag2, tag3) for tag1 in tag_list for tag2 in tag_list for tag3 in tag_list]
            for tag_pair in tag_pairs:
                N, V, D = tag_pair
                if (N, V, D) in nomin_count:
                    transitions[(N, V, D)] = np.log((nomin_count[(N, V, D)] + 1) / (denom_count[(V, D)] + len(denom_count)))
                else:
                    transitions[(N, V, D)] = np.log(1 / (denom_count[(V, D)] + len(denom_count)))
        return transitions         

    def _emisson_add_one(self, train_x, train_y):
        emissions = {}
        emission_count = defaultdict(int)
        tag_count = defaultdict(int)
        word_set = set()
        for sentence, tags in zip(train_x, train_y):
            for word, tag in zip(sentence, tags):
                if word == '*':
                    continue
                if tag == 'STOP':
                    break
                emission_count[(word, tag)] += 1
                tag_count[tag] += 1
                word_set.add(word)
        pairs = [(word, tag) for word in list(word_set) for tag in tag_count.keys()]
        for pair in pairs:
            word, tag = pair
            if (word, tag) in emission_count:
                emissions[(word, tag)] = np.log((emission_count[(word, tag)] + 1) / (tag_count[tag] + len(tag_count)))
            else:
                emissions[(word, tag)] = np.log(1 / (tag_count[tag] + len(tag_count)))
        return emissions

    def inference(self, x, decode):
        if decode == 'greedy':
            pass
        elif decode == 'beam':
            pass
        elif decode == 'viterbi':
            pass
        else:
            raise NotImplementedError('Decode method not implemented.')

    def _greedy(self, x):
        pass

    def _viterbi(self, x):
        pass

    def accuracy(self, dev_x, dev_y):
        pass

if __name__ == '__main__':
    ngram = 2
    smooth = 'add_one' # ['add_one', 'linear_interpolate']
    decode = 'viterbi' # ['greedy', 'beam', 'viterbi']

    loader = Loader(ngram=ngram)
    train_x, train_y = loader.load_data('train')
    dev_x, dev_y = loader.load_data('dev')
    test_x, _ = loader.load_data('test')
    common_set = loader.common_set
    rare_set = loader.rare_set

    hmm = HMM()
    hmm.train(train_x, train_y, ngram=ngram, smooth=smooth)
    hmm.inference(dev_x, decode=decode)

