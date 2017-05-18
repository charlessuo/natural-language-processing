import numpy as np
import gensim
import csv

embed_path = '/Users/howard50b/Documents/machine-learning/code/word-embedding/glove_6B_100d.txt'
model = gensim.models.Word2Vec.load_word2vec_format(embed_path, binary=False)
embed_size = len(model['the'])
print('Done loading embedding model.')


class Loader:
    def __init__(self, suffix_size=3):
        self.data_path = './data/{}_x.csv'
        self.label_path = './data/{}_y.csv'
        self.id_to_class = None
        self.class_to_id = None
        self.max_len = None

    def load_data(self, mode):
        sentences, labels = self._build_raw_sentences(mode)
        if mode == 'train':
            self.class_to_id, self.id_to_class = self._encode_labels(labels)
        x, y, seq_len = self._load_embedding(sentences, labels)
        return x, y, seq_len

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

    def _encode_labels(self, labels):
        class_to_id = {}
        idx = 1
        for tags in labels:
            for tag in tags:
                if tag not in class_to_id:
                    class_to_id[tag] = idx
                    idx += 1
        class_to_id['<PAD>'] = 0
        id_to_class = {v: k for k, v in class_to_id.items()}
        return class_to_id, id_to_class

    def _load_embedding(self, sentences, labels):
        x = np.zeros((len(sentences), self.max_len, embed_size))
        y = np.zeros((len(sentences), self.max_len))
        seq_len = []
        for i in range(len(sentences)):
            sentence_len = len(sentences[i])
            seq_len.append(sentence_len)
            for j in range(0, sentence_len):
                if sentences[i][j] in model:
                    x[i, j] = model[sentences[i][j]]
                else:
                    x[i, j] = np.random.rand(embed_size) - 0.5
            for j in range(sentence_len, self.max_len):
                x[i, j] = np.zeros(embed_size)

        for i, tags in enumerate(labels):
            for j, tag in enumerate(tags):
                y[i, j] = self.class_to_id[tag]
        return x, y, np.array(seq_len)


if __name__ == '__main__':
    loader = Loader()
    train_x, train_y, train_seq_len = loader.load_data('train')
    dev_x, dev_y, dev_seq_len = loader.load_data('dev')
    test_x, _, test_seq_len = loader.load_data('test')
    print('Max length in training data:', loader.max_len)
    print('Train data shape:', train_x.shape)
    print('Train label shape:', train_y.shape)
    print('Train max length:', max(train_seq_len))
    print('Dev   data shape:', dev_x.shape)
    print('Dev   label shape:', dev_y.shape)
    print('Dev max length:', max(dev_seq_len))
    print('Test  data shape:', test_x.shape)
    print('Test max length:', max(test_seq_len))


