#!/usr/bin/env python3
import numpy as np
from hmm_loader import Loader
from collections import defaultdict


class HMM:
    def __init__(self, tag_vocab):
        self.emissions = None
        self.transitions = None
        self.tag_list = list(tag_vocab)

        # self.states contains two-tag states (trigram)
        self.states = [(N, V) for N in self.tag_list 
                              for V in self.tag_list]

    def train(self, train_x, train_y, smooth, lambdas=None):
        '''
        Train HMM with ML estimations. 
        Store results in self.emissions and self.transitions.
        Args:
            train_x: List[List[str]], observations
            train_y: List[List[str]], tags
            smooth: str, smoothing method for transition estimations
                    'add_one' or 'linear_interpolate'
        '''
        self.emissions = self._emisson_probs(train_x, train_y)
        self.transitions = self._transition_probs(train_y, smooth, lambdas)

    def _emisson_probs(self, train_x, train_y):
        '''
        Calculate ML estimations to construct emission probability table.
        Args: 
            train_x: List[List[str]], observations
            train_y: List[List[str]], tags
        Returns:
            emissions: dict[(str, str)], emission probabilities, i.e. P(word | tag)
        '''
        emissions = defaultdict(lambda: float('-inf'))
#        emissions = {}
        emission_count = defaultdict(int)
        tag_count = defaultdict(int)
        for sentence, tags in zip(train_x, train_y):
            for word, tag in zip(sentence, tags):
                if tag == '*':
                    continue
                emission_count[(word, tag)] += 1
                tag_count[tag] += 1
        for word, tag in emission_count.keys():
            # only count existing probabilities
            # for later use, missing P(word | tag) will be viewed as -inf in log space
            emissions[(word, tag)] = np.log(emission_count[(word, tag)] / tag_count[tag])
        return emissions

    def _transition_probs(self, train_y, smooth, lambdas):
        if smooth == 'add_one':
            return self._transition_add_one(train_y)
        elif smooth == 'linear_interpolate':
            assert lambdas is not None
            return self._transition_linear_interpolate(train_y, lambdas)

    def _transition_linear_interpolate(self, train_y, lambdas):
        # TODO: check if it works correctly
        transitions = defaultdict(float)
        nomin2 = defaultdict(int)
        denom2 = defaultdict(int)
        nomin1 = defaultdict(int)
        denom1 = defaultdict(int)
        nomin0 = defaultdict(int)
        for tags in train_y:
            for i, tag in enumerate(tags):
                if tag == '*':
                    continue
                N, V, D = tags[i], tags[i - 1], tags[i - 2]
                nomin2[(N, V, D)] += 1
                denom2[(V, D)] += 1
                nomin1[(N, V)] += 1
                denom1[V] += 1
                nomin0[N] += 1
        tag_triplets = [(N, V, D) for N in self.tag_list 
                                  for V in self.tag_list 
                                  for D in self.tag_list]
        for triplet in tag_triplets:
            N, V, D = triplet
            if (N, V, D) in nomin2:
                transitions[(N, V, D)] += lambdas[2] * np.log(nomin2[(N, V, D)] / denom2[(V, D)])
            if (N, V) in nomin1:
                transitions[(N, V, D)] += lambdas[1] * np.log(nomin1[(N, V)] / denom1[V])
            if N in nomin0:
                transitions[(N, V, D)] += lambdas[0] * np.log(nomin0[N] / len(nomin0))
        return transitions 

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

    def inference(self, x, decode, k=None, verbose=False):
        if decode == 'beam':
            assert k is not None
            pred_y = self._beam(x, k, verbose)
        elif decode == 'viterbi':
            pred_y = self._viterbi(x, verbose)
        else:
            raise NotImplementedError('Decode method not implemented.')
        return pred_y

    def _beam(self, x, k, verbose):
        pred_y = []
        for c, sentence in enumerate(x):
            seqs = [['*', '*'] for _ in range(k)]
            total_scores = [0] * k
            for i, word in enumerate(sentence):
                if word == '*':
                    continue
                topk_scores = [float('-inf')] * k
                topk_backpointers = [(0, '')] * k # [(k_, N), ...]
                for k_ in range(k):
                    for state in self.states:
                        N, V = state
#                        if V != seqs[k_][-1] or (word, N) not in self.emissions:
#                            continue
                        if V != seqs[k_][-1] or self.emissions[(word, N)] == float('-inf'):
                            continue
                        D = seqs[k_][-2]
                        score = self.emissions[(word, N)] + self.transitions[(N, V, D)] + total_scores[k_]
                        min_score = min(topk_scores)
                        if score > min_score and score not in topk_scores and (k_, N) not in topk_backpointers:
                            min_idx = topk_scores.index(min_score)
                            topk_scores[min_idx] = score
                            topk_backpointers[min_idx] = (k_, N)
                new_seqs = []
                new_total_scores = []
                for score, bp in zip(topk_scores, topk_backpointers):
                    k_, N = bp
                    new_seqs.append(seqs[k_] + [N])
                    new_total_scores.append(score)
                seqs = new_seqs
                total_scores = new_total_scores
            if verbose and c % 50 == 0:
                print('%dth sentence:' % c)
                for seq in seqs:
                    print(seq)
            max_score_idx = total_scores.index(max(total_scores))
            pred_y.append(seqs[max_score_idx])
        return pred_y

    def _viterbi(self, x, verbose):
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
#                    if (word, N) not in self.emissions:
#                        new_pi[(N, V)] = float('-inf')
#                        continue
                    if self.emissions[(word, N)] == float('-inf'):
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
            if verbose and c % 50 == 0:
                print('%dth sentence:' % c)
                print(seq[::-1])
        return pred_y

    def accuracy(self, dev_x, dev_y, decode, k=None, verbose=False):
        pred_y = self.inference(dev_x, decode, k, verbose)
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

    def find_suboptimal_sequences(self, x, y, decode, k=None, verbose=False):
        pred_y = self.inference(x, decode, k, verbose)
        num_suboptimal = 0
        num_completely_correct = 0
        for sentence, pred_seq, gold_seq in zip(x, pred_y, y):
            pred_score = 0
            gold_score = 0
            for i in range(len(sentence)):
                if gold_seq[i] == '*':
                    continue
                pred_score += (self.transitions[(pred_seq[i], pred_seq[i - 1], pred_seq[i - 2])] 
                              + self.emissions[(sentence[i], pred_seq[i])])
                gold_score += (self.transitions[(gold_seq[i], gold_seq[i - 1], gold_seq[i - 2])]
                              + self.emissions[(sentence[i], gold_seq[i])])
            if gold_score > pred_score:
                num_suboptimal += 1
#                print('Predicted seq:', pred_seq)
#                print('Gold seq:     ', gold_seq)
            if gold_score == pred_score:
                num_completely_correct += 1
        return num_suboptimal / len(x), num_completely_correct / len(x)

def generate_submission(pred_sequences, filename='hmm_trigram_sample'):
    with open('./results/' + filename + '.csv', 'w') as f:
        f.write('id,tag\n')
        idx = 0
        for seq in pred_sequences:
            for tag in seq:
                if tag == '*' or tag == '<STOP>':
                    continue
                f.write('{},"{}"\n'.format(idx, tag))
                idx += 1


if __name__ == '__main__':
    loader = Loader(ngram=3)
    train_x, train_y = loader.load_data('train')
    dev_x, dev_y = loader.load_data('dev')
    test_x, _ = loader.load_data('test')
    print('Done loading data.')

    hmm = HMM(tag_vocab=loader.tag_vocab)
    # smooth = ['add_one', 'linear_interpolate']
#    hmm.train(train_x, train_y, smooth='linear_interpolate', lambdas=(0.6, 0.3, 0.1))
    hmm.train(train_x, train_y, smooth='add_one')
    print('Done training.')

    sample_size = 1000

    # inference
#    dev_acc = hmm.accuracy(dev_x[:sample_size], dev_y[:sample_size], decode='viterbi', verbose=False)
#    print('Dev accuracy (viterbi):', dev_acc)
#    dev_acc = hmm.accuracy(dev_x[:sample_size], dev_y[:sample_size], decode='beam', k=3, verbose=False)
#    print('Dev accuracy (beam):', dev_acc)

    # analysis
    viterbi_sub_rate, viterbi_correct_rate = hmm.find_suboptimal_sequences(dev_x[:sample_size], dev_y[:sample_size], decode='viterbi')
    print('Suboptimal sequence rate (viterbi):', viterbi_sub_rate)
    print('Correct sequence rate (viterbi):   ', viterbi_correct_rate)
    beam_sub_rate, beam_correct_rate = hmm.find_suboptimal_sequences(dev_x[:sample_size], dev_y[:sample_size], decode='beam', k=3)
    print('Suboptimal sequence rate (beam):', beam_sub_rate)
    print('Correct sequence rate (beam):   ', beam_correct_rate)

    # generate submission .csv file
#    pred_y = hmm.inference(test_x, decode='viterbi') # decode = ['beam', 'viterbi']
#    generate_submission(pred_y, filename='hmm_trigram_add_one_viterbi')

