import numpy as np
import csv


class Loader:
    def __init__(self, ngram):
        self.data_path = './data/{}_x.csv'
        self.label_path = './data/{}_y.csv'
        self.ngram = ngram
        self.common_set = None
        self.rare_set = None

    def load_data(self, mode):
        sentences, labels = self._build_raw_sentences(mode)
        if mode == 'train':
            counts = self._build_count_dict(sentences)
            self.common_set, self.rare_set = self._build_buckets(counts)
        sentences = self._build_data(sentences, mode)
        return sentences, labels

    def _build_raw_sentences(self, mode):
        ngram = self.ngram
        sentences = []
        labels = []
        if mode == 'train' or mode == 'dev':
            with open(self.data_path.format(mode)) as f_input, open(self.label_path.format(mode)) as f_label:
                next(f_input)
                next(f_label)
                sentence = ['*'] * (ngram - 1)
                tags = [None] * (ngram - 1)
                for input_line, label_line in zip(csv.reader(f_input), csv.reader(f_label)):
                    word = input_line[1]
                    tag = label_line[1]
                    sentence.append(word)
                    tags.append(tag)
                    if word == '.' or word == '?':
                        sentences.append(sentence + [None])
                        labels.append(tags + ['<STOP>'])
                        sentence = ['*'] * (ngram - 1)
                        tags = [None] * (ngram - 1)
        elif mode == 'test':
            with open(self.data_path.format(mode)) as f:
                next(f)
                sentence = ['*'] * (ngram - 1)
                for input_line in csv.reader(f):
                    word = input_line[1]
                    sentence.append(word)
                    if word == '.' or word == '?':
                        sentences.append(sentence + [None])
                        sentence = ['*'] * (ngram - 1)
        return sentences, labels

    def _build_count_dict(self, sentences):
        counts = {}
        for sentence in sentences:
            for word in sentence:
                if word == '*' or word is None:
                    continue
                if word not in counts:
                    counts[word] = 1
                else:
                    counts[word] += 1
        return counts

    def _build_buckets(self, counts):
        '''
        Build 3 buckets (sets) for common/rare/unseen words.
        Args:
            counts: dict, word-count mapping, words are all raw strings
        Returns:
            common_set: set, collection of words with count >= 5
            rare_set: set, collection of word suffixes with word 2 < count < 5
        '''
        common_set = set()
        rare_set = set()
        for word, count in counts.items():
            if count >= 5:
                common_set.add(word)
            if 2 < count < 5:
                rare_set.add(word[-2:])
        return common_set, rare_set

    def _build_data(self, sentences, mode):
        count0 = 0; count1 = 0; count2 = 0
        sentences_ = sentences[:]
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                if word == '*' or word is None:
                    continue
                suffix = word[-2:] # use suffix to collect unseen words
                if word in self.common_set:
                    sentences_[i][j] = word
                    count0 += 1
                elif suffix in self.rare_set:
                    sentences_[i][j] = suffix
                    count1 += 1
                else:
                    sentences_[i][j] = '<UNK>'
                    count2 += 1
        print('Mode %s' % mode)
        print('  Common words:', count0)
        print('  Rare words:', count1)
        print('  Unseen words:', count2)
        return sentences_


if __name__ == '__main__':
    loader = Loader(ngram=3)
    train_x, train_y = loader.load_data('train')
    dev_x, dev_y = loader.load_data('dev')
    test_x, _ = loader.load_data('test')

