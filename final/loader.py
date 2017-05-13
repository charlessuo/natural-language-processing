import numpy as np
import nltk


class Loader:
    def __init__(self):
        self.data_path = './data/{0}/{0}.txt'
        self.test_path = './data/test/test.nolabels.txt'
        self.word_to_id = None
        self.class_to_id = None
        self.pos_to_id = None
        self.id_to_word = None
        self.id_to_class = None
        self.id_to_pos = None
        self.max_len = 52
        self.common_set = set()

    def load_data(self, mode):
        sentences, pos_tags, labels = self._build_raw_sentences(mode)
        # only build vocabulary with training data
        if mode == 'train':
            counts = self._build_count_dict(sentences)
            self._build_buckets(counts)
            self._build_vocabs(counts, labels, pos_tags)
        x, pos, y = self._build_padded_data(mode, sentences, labels, pos_tags)
        return x, pos, y

    def _build_raw_sentences(self, mode):
        sentences = []
        labels = []
        pos_tags = []
        if mode == 'train' or mode == 'dev':
            with open(self.data_path.format(mode)) as f:
                sentence = []
                tags = []
                pos_tags = []
                for line in f:
                    line = line.split()
                    if not line:
                        sentences.append(sentence)
                        pos_tags.append([pair[1] for pair in 
                                        nltk.pos_tag(sentence)])
                        labels.append(tags)
                        sentence = []
                        tags = []
                        continue
                    word, tag = line
                    sentence.append(word)
                    tags.append(tag)
        elif mode == 'test':
            with open(self.test_path) as f:
                sentence = []
                for line in f:
                    line = line.strip()
                    if not line:
                        sentences.append(sentence)
                        pos_tags.append([pair[1] for pair in 
                                        nltk.pos_tag(sentence)])
                        sentence = []
                        continue
                    word = line
                    sentence.append(word)
        return sentences, pos_tags, labels

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
            if count > 0:
                self.common_set.add(word)

    def _build_vocabs(self, counts, labels, pos_tags):
        # build word-id mapping (frequent word has smaller index)
        sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        id_to_word = {}
        idx = 1
        for word, count in sorted_items:
            if word in self.common_set:
                id_to_word[idx] = word
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

        pos_to_id = {'NULL': 0}
        idx = 1
        for _pos_tags in pos_tags:
            for pos_tag in _pos_tags:
                if pos_tag not in pos_to_id:
                    pos_to_id[pos_tag] = idx
                    idx += 1
        id_to_pos = {v: k for k, v in pos_to_id.items()}
        self.pos_to_id = pos_to_id
        self.id_to_pos = id_to_pos

    def _build_padded_data(self, mode, sentences, labels_, pos_tags):
        bucket_counts = {'common': 0, 'rare': 0, 'unk': 0}
        inputs = np.zeros((len(sentences), self.max_len)).astype(int)
        pos_inputs = np.zeros(inputs.shape).astype(int)
        labels = np.zeros(inputs.shape).astype(int)
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                if word in self.word_to_id:
                    inputs[i, j] = self.word_to_id[word]
                    bucket_counts['common'] += 1
                else:
                    inputs[i, j] = self.word_to_id['<UNK>']
                    bucket_counts['unk'] += 1
        print('Mode: {}, {}'.format(mode, bucket_counts))
        for i, _pos_tags in enumerate(pos_tags):
            for j, pos_tag in enumerate(_pos_tags):
                if pos_tag in self.pos_to_id:
                    pos_inputs[i, j] = self.pos_to_id[pos_tag]
                else:
                    pos_inputs[i, j] = 0

        # test mode only has inputs
        if mode == 'test':
            return inputs, pos_inputs, None

        for i, sentence_tags in enumerate(labels_):
            for j, tag in enumerate(sentence_tags):
                labels[i, j] = self.class_to_id[tag]
        return inputs, pos_inputs, labels


if __name__ == '__main__':
    loader = Loader()
    train_x, train_x_pos, train_y = loader.load_data('train')
    dev_x, dev_x_pos, dev_y = loader.load_data('dev')
    test_x, test_x_pos, _ = loader.load_data('test')
    print('Max length in training data:', loader.max_len)
    print('Train tokens:', np.sum(train_x > 0))
    print('Train sentences:', len(train_x))
    print('Dev tokens:', np.sum(dev_x > 0))
    print('Dev sentences:', len(dev_x))
    print('Test tokens:', np.sum(test_x > 0))
    print('Test sentences:', len(test_x))

