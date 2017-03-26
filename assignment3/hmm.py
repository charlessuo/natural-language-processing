#!/usr/bin/env python3
import numpy as np


class HMM:
    def __init__(self):
        self.emissions = {}
        self.transmissions = {}
        self.states = {}

    def train(self):
        pass

    def inference(self):
        pass


if __name__ == '__main__':
    loader = Loader()
    train_x, train_y = loader.load('train')
    dev_x, dev_y = loader.load('dev')
    test_x, _ = loader.load('test')

    id_to_word = loader.id_to_word
    id_to_class = loader.id_to_class
    hmm = HMM()

