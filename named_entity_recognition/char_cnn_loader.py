import numpy as np


class CharCNNLoader:
    def __init__(self):
        self.data_path = './data/{0}/{0}.txt'
        self.test_path = './data/test/test.nolabels.txt'
        self.char_to_id = None
        self.id_to_char = None
        self.max_len = 52
        self.char_len = 20

    def load_data(self, mode):
        sentences = self._build_raw_sentences(mode)
        # only build vocabulary with training data
        if mode == 'train':
            self._build_char_vocabs(sentences)
        x = self._build_padded_data(sentences)
        return x

    def _build_raw_sentences(self, mode):
        sentences = []
        labels = []
        pos_tags = []
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

    def _build_char_vocabs(self, sentences):
        char_to_id = {}
        idx = 1
        for sentence in sentences:
            for word in sentence:
                for char in word:
                    if char not in char_to_id:
                        char_to_id[char] = idx
                        idx += 1
        char_to_id['<PAD>'] = 0
        char_to_id['<UNK>'] = idx
        id_to_char = {v: k for k, v in char_to_id.items()}
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char

    def _build_padded_data(self, sentences):
        inputs = np.zeros((len(sentences), self.max_len, self.char_len)).astype(int)
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                for k, char in enumerate(word[:self.char_len]):
                    if char in self.char_to_id:
                        inputs[i, j, k] = self.char_to_id[char]
                    else:
                        inputs[i, j, k] = self.char_to_id['<UNK>']
        return inputs


if __name__ == '__main__':
    loader = CharCNNLoader()
    train_char = loader.load_data('train')
    dev_char = loader.load_data('dev')
    test_char = loader.load_data('test')
    print('Train tokens:', np.sum(train_char > 0))
    print('Train sentences:', len(train_char))
    print('Dev tokens:', np.sum(dev_char > 0))
    print('Dev sentences:', len(dev_char))
    print('Test tokens:', np.sum(test_char > 0))
    print('Test sentences:', len(test_char))

