#!/usr/bin/env python3
import numpy as np
from hmm_loader import Loader


class HMM:
    def __init__(self):
        self.emissions = {}
        self.transitions = {}
        self.states = {}

    def train(self, train_x, train_y):
        pass

    def _transition_probs(self, train_x):
        pass
        
    def _emisson_probs(self, train_x, train_y):
        for sentence, tags in zip(train_x, train_y):
            for word, tag in zip(sentence, tags):
                if word == 0:
                    # when <PAD> is seen, break to the next line
                    break
                if (word, tag) not in self.emissions:
                    self.emissions[(word, tag)] = 1
                else:
                    self.emissions[(word, tag)] += 1

    def inference(self, x, y=None, mode='viterbi'):
        pass


if __name__ == '__main__':
    loader = Loader()
    train_x, train_y = loader.load_data('train')
    dev_x, dev_y = loader.load_data('dev')
    test_x, _ = loader.load_data('test')

    id_to_word = loader.id_to_word
    id_to_class = loader.id_to_class
    hmm = HMM()

