import numpy as np
import csv


class Loader:
    def __init__(self, ngram, suffix_size=3):
        self.data_path = './data/{}_x.csv'
        self.label_path = './data/{}_y.csv'
        self.ngram = ngram
        self.tag_vocab = set(['*', '<STOP>'])
        self.common_set = None
        self.rare_set = None
        self.suffix_size = suffix_size

    def load_data(self, mode):
        '''
        Load data.
        Args:
            mode: str, 'train', 'dev', or 'test'
        Returns:
            sentences: List[List[str]], final sentences
            labels: List[List[str]], final labels
        '''
        sentences, labels = self._build_raw_sentences(mode)
        if mode == 'train':
            counts = self._build_count_dict(sentences)
            self.common_set, self.rare_set = self._build_buckets(counts)
        sentences = self._build_data(sentences, mode)
        return sentences, labels

    def _build_raw_sentences(self, mode):
        '''
        Load raw sentences and labels from file
        and add paddings e.g. '*' and '<STOP>'.
        Args:
            mode: str, 'train', 'dev', or 'test'
        Returns:
            sentences: List[List[str]], raw sentences with paddings
            labels: List[List[str]], raw labels with paddings
        '''
        ngram = self.ngram
        sentences = []
        labels = []
        if mode == 'train' or mode == 'dev':
            with open(self.data_path.format(mode)) as f_input, open(self.label_path.format(mode)) as f_label:
                next(f_input)
                next(f_label)
                sentence = ['*'] * (ngram - 1)
                tags = ['*'] * (ngram - 1)
                for input_line, label_line in zip(csv.reader(f_input), csv.reader(f_label)):
                    word = input_line[1]
                    tag = label_line[1]
                    sentence.append(word)
                    tags.append(tag)
                    self.tag_vocab.add(tag) # build vocab for tags
                    if word == '.' or word == '?':
                        sentences.append(sentence + ['<STOP>'])
                        labels.append(tags + ['<STOP>'])
                        sentence = ['*'] * (ngram - 1)
                        tags = ['*'] * (ngram - 1)
        elif mode == 'test':
            with open(self.data_path.format(mode)) as f:
                next(f)
                sentence = ['*'] * (ngram - 1)
                for input_line in csv.reader(f):
                    word = input_line[1]
                    sentence.append(word)
                    if word == '.' or word == '?':
                        sentences.append(sentence + ['<STOP>'])
                        sentence = ['*'] * (ngram - 1)
        return sentences, labels

    def _build_count_dict(self, sentences):
        counts = {}
        for sentence in sentences:
            for word in sentence:
                if word == '*' or word == '<STOP>':
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
            common_set: set, collection of words with count > 2
            rare_set: set, collection of word suffixes with word count == 2
        '''
        common_set = set()
        rare_set = set()
        num_set = set()
        for word, count in counts.items():
            if count > 2:
                common_set.add(word)
            if count == 2:
                rare_set.add(word[-self.suffix_size:])
        return common_set, rare_set

    def _build_data(self, sentences, mode):
        '''
        Replace words in sentences with its suffix/'<UNK>' token when 
        it's in rare_set/unseen_set.
        Args:
            sentences: List[List[str]], raw sentences with paddings
            mode: str, 'train', 'dev', or 'test'
        Returns:
            sentences_: List[List[str]], final sentences
        '''
        bucket_counts = {'common': 0, 'rare': 0, 'unk': 0}
        sentences_ = sentences[:]
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                if word == '*' or word == '<STOP>':
                    continue
                suffix = word[-self.suffix_size:] # use suffix to collect unseen words
                if word in self.common_set:
                    sentences_[i][j] = word
                    bucket_counts['common'] += 1
                elif suffix in self.rare_set:
                    sentences_[i][j] = suffix
                    bucket_counts['rare'] += 1
                else:
                    sentences_[i][j] = '<UNK>'
                    bucket_counts['unk'] += 1
        print('Mode %s' % mode)
        print('  Common words:', bucket_counts['common'])
        print('  Rare words:', bucket_counts['rare'])
        print('  Unseen words:', bucket_counts['unk'])
        return sentences_


if __name__ == '__main__':
    loader = Loader(ngram=3)
    train_x, train_y = loader.load_data('train')
    dev_x, dev_y = loader.load_data('dev')
    test_x, _ = loader.load_data('test')

