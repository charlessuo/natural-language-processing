import numpy as np
import csv


class Loader:
    def __init__(self):
        self.data_path = './data/{}_x.csv'
        self.label_path = './data/{}_y.csv'
        self.word_to_id = None
        self.class_to_id = None
        self.id_to_word = None
        self.id_to_class = None
        self.max_len = None
        self.common_set = set()
        self.rare_set = set()

    def load_data(self, mode):
        sentences, labels = self._build_raw_sentences(mode)
        # only build vocabulary with training data
        if mode == 'train':
            counts = self._build_count_dict(sentences)
            self._build_buckets(counts)
            self._build_vocabs(counts, labels)
        x, y = self._build_padded_data(mode, sentences, labels)
        return x, y

    def _build_raw_sentences(self, mode):
        sentences = []
        labels = []
        max_len = 0
        if mode == 'train' or mode == 'dev':
            with open(self.data_path.format(mode)) as f_input, open(self.label_path.format(mode)) as f_label:
                next(f_input)
                next(f_label)
                sentence = []
                tags = []
                for input_line, label_line in zip(csv.reader(f_input), csv.reader(f_label)):
                    word = input_line[1]
                    tag = label_line[1]
                    sentence.append(word)
                    tags.append(tag)
                    if word == '.' or word == '?':
                        max_len = max(max_len, len(sentence))
                        sentences.append(sentence)
                        labels.append(tags)
                        sentence = []
                        tags = []
            if mode == 'train':
                self.max_len = max_len
        elif mode == 'test':
            with open(self.data_path.format(mode)) as f:
                next(f)
                sentence = []
                for input_line in csv.reader(f):
                    word = input_line[1]
                    sentence.append(word)
                    if word == '.' or word == '?':
                        sentences.append(sentence)
                        sentence = []
        return sentences, labels

    def _build_count_dict(self, sentences):
        counts = {}
        for sentence in sentences:
            for word in sentence:
                if word not in counts:
                    counts[word] = 1
                else:
                    counts[word] += 1
        return counts

    def _build_buckets(self, counts):
        for word, count in counts.items():
            if count > 2:
                self.common_set.add(word)
            if count == 2:
                self.rare_set.add(word[-2:])

    def _build_vocabs(self, counts, labels):
        # build word-id mapping (frequent word has smaller index)
        sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        id_to_word = {}
        idx = 1
        for word, count in sorted_items:
            if word in self.common_set:
                id_to_word[idx] = word
                idx += 1

        for suffix in list(self.rare_set):
            id_to_word[idx] = suffix
            idx += 1

        id_to_word[0] = '<PAD>'
        id_to_word[idx] = '<UNK>'
        word_to_id = {v: k for k, v in id_to_word.items()}
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        
        # build class-id mapping
        class_to_id = {}
        idx = 1
        for sentence_tags in labels:
            for tag in sentence_tags:
                if tag not in class_to_id:
                    class_to_id[tag] = idx
                    idx += 1
        class_to_id['<PAD>'] = 0
        id_to_class = {v: k for k, v in class_to_id.items()}
        self.class_to_id = class_to_id
        self.id_to_class = id_to_class

    def _build_padded_data(self, mode, sentences, labels_):
        bucket_counts = {'common': 0, 'rare': 0, 'unk': 0}
        inputs = np.zeros((len(sentences), self.max_len)).astype(int)
        labels = np.zeros(inputs.shape).astype(int)
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                suffix = word[-2:]
                if word in self.word_to_id:
                    inputs[i, j] = self.word_to_id[word]
                    bucket_counts['common'] += 1
                elif suffix in self.word_to_id:
                    inputs[i, j] = self.word_to_id[suffix]
                    bucket_counts['rare'] += 1
                else:
                    inputs[i, j] = self.word_to_id['<UNK>']
                    bucket_counts['unk'] += 1
        print('Mode: {}, {}'.format(mode, bucket_counts))

        # test mode only has inputs
        if mode == 'test':
            return inputs, None

        for i, sentence_tags in enumerate(labels_):
            for j, tag in enumerate(sentence_tags):
                labels[i, j] = self.class_to_id[tag]
        return inputs, labels


if __name__ == '__main__':
    loader = Loader()
    train_x, train_y = loader.load_data('train')
    dev_x, dev_y = loader.load_data('dev')
    test_x, _ = loader.load_data('test')
    print('Max length in training data:', loader.max_len)
    print('Train tokens:', np.sum(train_x > 0))
    print('Train sentences:', len(train_x))
    print('Dev tokens:', np.sum(dev_x > 0))
    print('Dev sentences:', len(dev_x))
    print('Test tokens:', np.sum(test_x > 0))
    print('Test sentences:', len(test_x))

