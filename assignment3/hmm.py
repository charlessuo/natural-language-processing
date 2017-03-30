#!/usr/bin/env python3
import numpy as np
from hmm_loader import Loader
from collections import defaultdict

class HMM:
    def __init__(self, tag_vocab):
        self.emissions = None
        self.transitions = None
        self.tag_list = list(tag_vocab)
        self.states = [(N, V) for N in self.tag_list 
                              for V in self.tag_list] 

    def train(self, train_x, train_y, smooth):
        self.emissions = self._emisson_probs(train_x, train_y)
        self.transitions = self._transition_probs(train_y, smooth)

    def _emisson_probs(self, train_x, train_y):
        emissions = {}
        emission_count = defaultdict(int)
        tag_count = defaultdict(int)
        for sentence, tags in zip(train_x, train_y):
            for word, tag in zip(sentence, tags):
                if tag == '*':
                    continue
#                if tag == '<STOP>':
#                    break
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
        nomin_count = defaultdict(int)
        denom_count = defaultdict(int)
        for tags in train_y:
            for i, tag in enumerate(tags):
                if tag == '*':
                    continue
                N, V, D = tags[i], tags[i - 1], tags[i - 2]
                nomin_count[(N, V, D)] += 1
                denom_count[(V, D)] += 1
        tag_triplets = [(N, V, D) for N in self.tag_list 
                                  for V in self.tag_list 
                                  for D in self.tag_list]
        for triplet in tag_triplets:
            N, V, D = triplet
            if (N, V, D) in nomin_count:
                transitions[(N, V, D)] = np.log((nomin_count[(N, V, D)] + 1) / (denom_count[(V, D)] + len(denom_count)))
            else:
                transitions[(N, V, D)] = np.log(1 / (denom_count[(V, D)] + len(denom_count)))
        return transitions 

    def inference(self, x, decode, k=None):
        if decode == 'beam':
            assert k is not None
            pred_y = self._beam(x, k)
        elif decode == 'viterbi':
            pred_y = self._viterbi(x)
        else:
            raise NotImplementedError('Decode method not implemented.')
        return pred_y

    def _beam(self, x, k):
        pred_y = []
        for sentence in x:
            seq = ['*', '*']
            total_score = 0
            for i, word in enumerate(sentence):
                if word == '*':
                    continue
                score = float('-inf')
                back_pointer = None
                for state in self.states:
                    N, V = state
                    if V != seq[-1] or (word, N) not in self.emissions:
                        continue
                    D = seq[-2]
                    score_ = self.emissions[(word, N)] + self.transitions[(N, V, D)]
                    if score_ > score:
                        score = score_
                        back_pointer = N
                total_score += score
                seq.append(back_pointer)
            pred_y.append(seq)
        return pred_y

    def _viterbi(self, x):
        pred_y = []
        for c, sentence in enumerate(x):
            pi = {state: float('-inf') for state in self.states}
            pi[('*', '*')] = 0
            back_pointer = {}
            bp_idx = 0
            for i, word in enumerate(sentence):
                if word == '*':
                    continue
                new_pi = {state: float('-inf') for state in self.states}
                for state in self.states:
                    N, V = state
                    if (word, N) not in self.emissions:
                        new_pi[(N, V)] = float('-inf')
                        continue
                    for D in self.tag_list:
                        score = pi[(V, D)] + self.transitions[(N, V, D)] + self.emissions[(word, N)]
                        if score > new_pi[(N, V)]:
                            new_pi[(N, V)] = score
                            back_pointer[(bp_idx, N, V)] = D # bp_idx starts from 0
                pi = new_pi
                bp_idx += 1
            # generate sequence from back pointer
            last_state = max(pi, key=pi.get) # get last state with the highest score
            seq = list(last_state)           # put last two tags into the sequence
            bp_idx -= 1                      # set back pointer index to the right ending point
            while bp_idx >= 0:
                N, V = seq[-2:]
                D = back_pointer[(bp_idx, N, V)]
                seq.append(D)
                bp_idx -= 1
            pred_y.append(seq[::-1])
            if c % 20 == 0:
                print('%dth sentence:' % c)
                print(seq[::-1])
        return pred_y

    def accuracy(self, dev_x, dev_y, decode, k=None):
        pred_y = self.inference(dev_x, decode, k)
        num_correct = 0
        total = 0
        for pred_seq, dev_seq in zip(pred_y, dev_y):
            for y_, y in zip(pred_seq, dev_seq):
                if y == '*' or y == '<STOP>':
                    continue
                if y_ == y:
                    num_correct += 1
                total += 1
        return num_correct / total

if __name__ == '__main__':

    loader = Loader(ngram=3)
    train_x, train_y = loader.load_data('train')
    dev_x, dev_y = loader.load_data('dev')
    test_x, _ = loader.load_data('test')
#    common_set = loader.common_set
#    rare_set = loader.rare_set

    hmm = HMM(tag_vocab=loader.tag_vocab)
    hmm.train(train_x, train_y, smooth='add_one') # smooth = ['add_one', 'linear_interpolate']

#    pred_y = hmm.inference(dev_x, decode='beam', k=1) # decode = ['beam', 'viterbi']
#    accuracy = hmm.accuracy(dev_x, dev_y, decode='beam', k=1)

#    pred_y = hmm.inference(dev_x, decode='viterbi') # decode = ['beam', 'viterbi']
    accuracy = hmm.accuracy(dev_x[:2000], dev_y[:2000], decode='viterbi')

    print(accuracy)

