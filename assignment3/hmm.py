#!/usr/bin/env python3
import numpy as np
from hmm_loader import Loader
from collections import defaultdict

class HMM:
    def __init__(self):
        self.emissions = None
        self.transitions = None
        self.states = {}

    def train(self, train_x, train_y, smooth):
        self.emissions = self._emisson_probs(train_x, train_y)
        self.transitions = self._transition_probs(train_y, smooth)

    def _emisson_probs(self, train_x, train_y):
        emissions = {}
        emission_count = defaultdict(int)
        tag_count = defaultdict(int)
        for sentence, tags in zip(train_x, train_y):
            for word, tag in zip(sentence, tags):
                if word == '*':
                    continue
                if tag == '<STOP>':
                    break
                emission_count[(word, tag)] += 1
                tag_count[tag] += 1
        for word, tag in emission_count.keys():
            emissions[(word, tag)] = np.log(emission_count[(word, tag)] / tag_count[tag])
#            if word == '<UNK>':
#                print(tag, emissions[(word, tag)])
        return emissions

    def _transition_probs(self, train_y, smooth):
        if smooth == 'add_one':
            return self._transition_add_one(train_y)
        elif smooth == 'linear_interpolate':
            return self._transition_linear_interpolate(train_y)

    def _transition_linear_interpolate(self, train_y):
        pass

    def _transition_add_one(self, train_y):
        transitions = {}
        tag_set = set()
        nomin_count = defaultdict(int)
        denom_count = defaultdict(int)
        for tags in train_y:
            for i, tag in enumerate(tags):
                if tag is None:
                    continue
                tag_set.add(tag)
                N, V, D = tags[i], tags[i - 1], tags[i - 2]
                nomin_count[(N, V, D)] += 1
                denom_count[(V, D)] += 1
        tag_list = list(tag_set)

        tag_pairs = [(tag1, tag2, tag3) for tag1 in tag_list for tag2 in tag_list for tag3 in tag_list]
        for tag_pair in tag_pairs:
            N, V, D = tag_pair
            if (N, V, D) in nomin_count:
                transitions[(N, V, D)] = np.log((nomin_count[(N, V, D)] + 1) / (denom_count[(V, D)] + len(denom_count)))
            else:
                transitions[(N, V, D)] = np.log(1 / (denom_count[(V, D)] + len(denom_count)))
        return transitions         

    def inference(self, x, decode, k=None):
        if decode == 'greedy':
            y_ = self._greedy(x)
        elif decode == 'beam':
            assert k is not None
            y_ = self.beam(x, k)
        elif decode == 'viterbi':
            y_ = self._viterbi(x)
        else:
            raise NotImplementedError('Decode method not implemented.')
        return y_

    def _greedy(self, x):
        y_ = []
        return y_

    def _viterbi(self, x):
        y_ = []
        return y_

    def accuracy(self, dev_x, dev_y):
        pass

if __name__ == '__main__':

    loader = Loader(ngram=3)
    train_x, train_y = loader.load_data('train')
    dev_x, dev_y = loader.load_data('dev')
    test_x, _ = loader.load_data('test')
    common_set = loader.common_set
    rare_set = loader.rare_set

    hmm = HMM()
    hmm.train(train_x, train_y, smooth='add_one')
    y_ = hmm.inference(dev_x, decode='viterbi') # decode = ['greedy', 'beam', 'viterbi']

