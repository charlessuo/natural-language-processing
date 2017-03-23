import numpy as np
import csv


class Loader:
    def __init__(self):
        self.train_x_path = './data/train_x.csv'
        self.train_y_path = './data/train_y.csv'

    def load(self, mode):
        if mode == 'train':
            sentences, labels, max_len = self._load_sentences(self.train_x_path, self.train_y_path)
            counts = self._build_count_dict(sentences)
            word_to_id, id_to_word, class_to_id, id_to_class = self._build_vocabs(counts, labels)
            train_x, train_y = self._build_padded_data(sentences, labels, max_len, word_to_id, class_to_id) 
            return train_x, train_y, max_len, id_to_word, id_to_class
        elif mode == 'dev':
            pass
        elif mode == 'test':
            pass

    def _load_sentences(self, input_path, label_path):
        sentences = []
        labels = []
        max_len = 0
        with open(input_path) as f_input, open(label_path) as f_label:
            next(f_input)
            next(f_label)
            sentence = []
            tags = []
            for input_line, label_line in zip(csv.reader(f_input), csv.reader(f_label)):
                word = input_line[1]
                tag = label_line[1]
                if tag == '.':
                    max_len = max(max_len, len(sentence))
                    sentences.append(sentence)
                    labels.append(tags)
                    sentence = []
                    tags = []
                else:
                    sentence.append(word)
                    tags.append(tag)
        return sentences, labels, max_len

    def _build_count_dict(self, sentences):
        counts = {}
        for sentence in sentences:
            for word in sentence:
                if word not in counts:
                    counts[word] = 1
                else:
                    counts[word] += 1
        return counts

    def _build_vocabs(self, counts, labels):
        # build word-id mapping (frequent word has smaller index)
        sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        id_to_word = {}
        idx = 1
        for id_, count in sorted_items:
            id_to_word[idx] = id_
            idx += 1 
        id_to_word[0] = '<PAD>'
        id_to_word[idx] = '<UNK>'
        word_to_id = {v: k for k, v in id_to_word.items()}
        
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

        return word_to_id, id_to_word, class_to_id, id_to_class

    def _build_padded_data(self, sentences, labels_, max_len, word_to_id, class_to_id):
        inputs = np.zeros((len(sentences), max_len)).astype(int)
        labels = np.zeros(inputs.shape).astype(int)
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                inputs[i, j] = word_to_id[word]

        for i, sentence_tags in enumerate(labels_):
            for j, tag in enumerate(sentence_tags):
                labels[i, j] = class_to_id[tag]
        return inputs, labels


if __name__ == '__main__':
    loader = Loader()
    train_x, train_y, max_len, id_to_word, id_to_class = loader.load('train')
    print(train_x[:5]) 
    print(train_y[:5])

