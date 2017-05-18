import numpy as np


class OrthLoader:
    def __init__(self):
        self.data_path = './data/{0}/{0}.txt'
        self.test_path = './data/test/test.nolabels.txt'
        self.word_to_id = None
        self.id_to_word = None
        self.max_len = 52

    def load_data(self, mode):
        sentences = self._build_raw_sentences(mode)
        # only build vocabulary with training data
        if mode == 'train':
            counts = self._build_count_dict(sentences)
            self._build_vocabs(counts)
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
                    sentence.append(self._orthographic(word))
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
                    sentence.append(self._orthographic(word))
        return sentences

    def _orthographic(self, word):
        ortho_word = ''
        for char in word:
            if char.isupper():
                ortho_word += 'C'
            elif char.islower():
                ortho_word += 'c'
            elif char.isdigit():
                ortho_word += 'n'
            else:
                ortho_word += 'p'
        return ortho_word

    def _build_count_dict(self, sentences):
        counts = {}
        for sentence in sentences:
            for word in sentence:
                if word not in counts:
                    counts[word] = 1
                else:
                    counts[word] += 1
        return counts

    def _build_vocabs(self, counts):
        # build word-id mapping (frequent word has smaller index)
        sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        word_to_id = {}
        idx = 1
        for word, count in sorted_items:
            if word not in word_to_id:
                word_to_id[word] = idx
                idx += 1
        word_to_id['<PAD>'] = 0
        word_to_id['<UNK>'] = idx
        id_to_word = {v: k for k, v in word_to_id.items()}
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

    def _build_padded_data(self, sentences):
        inputs = np.zeros((len(sentences), self.max_len)).astype(int)
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                if word in self.word_to_id:
                    inputs[i, j] = self.word_to_id[word]
                else:
                    inputs[i, j] = self.word_to_id['<UNK>']
        return inputs


if __name__ == '__main__':
    loader = OrthLoader()
    train_orth = loader.load_data('train')
    dev_orth = loader.load_data('dev')
    test_orth = loader.load_data('test')
#    print([loader.id_to_word[_] for _ in train_orth[0]])
    print('Max length in training data:', loader.max_len)
    print('Train tokens:', np.sum(train_orth > 0))
    print('Train sentences:', len(train_orth))
    print('Dev tokens:', np.sum(dev_orth > 0))
    print('Dev sentences:', len(dev_orth))
    print('Test tokens:', np.sum(test_orth > 0))
    print('Test sentences:', len(test_orth))

