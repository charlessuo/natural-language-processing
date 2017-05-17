import numpy as np


class ClusterLoader:
    def __init__(self):
        self.data_path = './data/{0}/{0}.txt'
        self.test_path = './data/test/test.nolabels.txt'
        self.cluster_path = './data/allwords_cluster'
        self.word_to_prefix = None
        self.cluster_vocab_size = None
        self.max_len = 52

    def load_data(self, mode):
        sentences = self._build_raw_sentences(mode)
        # only build vocabulary with training data
        if mode == 'train':
            self._build_cluster_vocab()
        x = self._build_padded_data(sentences)
        return x

    def _build_cluster_vocab(self):
        word_to_prefix = {}
        prefix_to_id = {}
        with open(self.cluster_path) as f:
            idx = 0
            for line in f:
                path_str, word, _ = line.split()
                prefix = path_str[:10]
                if prefix in prefix_to_id:
                    word_to_prefix[word] = prefix_to_id[prefix]
                else:
                    prefix_to_id[prefix] = idx
                    word_to_prefix[word] = prefix_to_id[prefix]
                    idx += 1
        self.word_to_prefix = word_to_prefix
        self.cluster_vocab_size = len(set(word_to_prefix.values()))

    def _build_raw_sentences(self, mode):
        sentences = []
        if mode == 'train' or mode == 'dev':
            with open(self.data_path.format(mode)) as f:
                sentence = []
                for line in f:
                    line = line.split()
                    if not line:
                        sentences.append(sentence)
                        sentence = []
                        continue
                    word, _ = line
                    sentence.append(word)
        elif mode == 'test':
            with open(self.test_path) as f:
                sentence = []
                for line in f:
                    line = line.strip()
                    if not line:
                        sentences.append(sentence)
                        sentence = []
                        continue
                    word = line
                    sentence.append(word)
        return sentences

    def _build_padded_data(self, sentences):
        inputs = np.zeros((len(sentences), self.max_len, self.cluster_vocab_size)).astype(float)
        one_hot = np.eye(self.cluster_vocab_size)
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                inputs[i, j] = one_hot[self.word_to_prefix[word]]
        return inputs


if __name__ == '__main__':
    loader = ClusterLoader()
    train_data = loader.load_data('train')
    dev_data = loader.load_data('dev')
    test_data = loader.load_data('test')
    print('Train tokens:', np.sum(train_data > 0))
    print('Train sentences:', len(train_data))
    print('Dev tokens:', np.sum(dev_data > 0))
    print('Dev sentences:', len(dev_data))
    print('Test tokens:', np.sum(test_data > 0))
    print('Test sentences:', len(test_data))

