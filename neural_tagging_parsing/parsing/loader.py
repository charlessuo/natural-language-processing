import numpy as np
from collections import defaultdict, deque
import time


class Loader:
    def __init__(self):
        self.data_path = './data/{}.conll'
        self.word_vocab = None
        self.tag_vocab = None
        self.label_vocab = None
        self.rev_word_vocab = None
        self.rev_tag_vocab = None
        self.rev_label_vocab = None
        self.action_space = None
        self.rev_action_space = None

    def load_train_data(self):
        sentences, children_list = self._load_data('train')
        self.word_vocab, self.tag_vocab, self.label_vocab = \
            self._build_vocab(sentences)
        self.rev_word_vocab = {v: k for k, v in self.word_vocab.items()}
        self.rev_tag_vocab = {v: k for k, v in self.tag_vocab.items()}
        self.rev_label_vocab = {v: k for k, v in self.label_vocab.items()}
        confs, actions = self._build_train_pairs(sentences, children_list)
        return confs, actions

    def load_dev_data(self):
        sentences, children_list = self._load_data('dev')
        return sentences

    def load_test_data(self):
        sentences, children_list = self._load_data('test')
        return sentences
    
    def _load_data(self, mode):
        sentences = []
        children_list = []
        with open(self.data_path.format(mode)) as f:
            sentence = []
            children = defaultdict(int)
            for line in f:
                line = line.split()
                if not line:
                    sentences.append(sentence)
                    sentence = []
                    children_list.append(children)
                    children = defaultdict(int)
                    continue
                id_, word, _, tag, tag, _, head, label, _, _ = line
                children[head] += 1 
                sentence.append([word, tag, label, id_, head])
        return sentences, children_list

    def _build_vocab(self, sentences):
        word_vocab = {'<UNK>': 0, '<ROOT>': 1}
        tag_vocab = {'NULL': 0}
        label_vocab = {'ROOT': 0, 'NULL': 1}
        word_idx = 2 
        tag_idx = 1
        label_idx = 2
        for sentence in sentences:
            for item in sentence:
                word, tag, label, _, _ = item
                if word not in word_vocab:
                    word_vocab[word] = word_idx
                    word_idx += 1
                if tag not in tag_vocab:
                    tag_vocab[tag] = tag_idx
                    tag_idx += 1
                if label not in label_vocab:
                    label_vocab[label] = label_idx
                    label_idx += 1
        return word_vocab, tag_vocab, label_vocab

    def _build_train_pairs(self, sentences, children_list):
        confs = []
        actions = []
        action_space = {}
        action_idx = 0
        for sentence, children in zip(sentences, children_list):
            stack = deque([['<ROOT>', 'NULL', 'ROOT', 0, None]])
            buffer_ = deque(sentence)
            while buffer_:
                s = stack.pop()
                s_word, s_tag, s_label, s_id, s_head = s
                b = buffer_.popleft()
                b_word, b_tag, b_label, b_id, b_head = b
                s_word_idx, s_tag_idx, s_label_idx = self._map_to_idx(s)
                b_word_idx, b_tag_idx, b_label_idx = self._map_to_idx(b)

                if b_id == s_head and children[s_id] == 0:
                    if ('LEFT_ARC', s_label) not in action_space:
                        action_space[('LEFT_ARC', s_label)] = action_idx
                        action_idx += 1
                    buffer_.appendleft(b)
                    children[b_id] -= 1
                    actions.append(action_space[('LEFT_ARC', s_label)])
                elif s_id == b_head and children[b_id] == 0:
                    if ('RIGHT_ARC', b_label) not in action_space:
                        action_space[('RIGHT_ARC', b_label)] = action_idx
                        action_idx += 1
                    buffer_.appendleft(s)
                    children[s_id] -= 1
                    actions.append(action_space[('RIGHT_ARC', b_label)])
                else:
                    if ('SHIFT', 'NULL') not in action_space:
                        action_space[('SHIFT', 'NULL')] = action_idx
                        action_idx += 1
                    stack.append(s)
                    stack.append(b)
                    actions.append(action_space[('SHIFT', 'NULL')])
                confs.append([s_word_idx, b_word_idx, 
                              s_tag_idx, b_tag_idx, 
                              s_label_idx, b_label_idx])

        if not self.action_space:
            self.action_space = action_space
            self.rev_action_space = {v: k for k, v in action_space.items()}
            
        return np.array(confs), np.array(actions)

    def _map_to_idx(self, node):
        word, tag, label, _, _ = node
        if word in self.word_vocab:
            word_idx = self.word_vocab[word]
        else:
            word_idx = 0
        tag_idx = self.tag_vocab[tag]
        label_idx = self.label_vocab[label]
        return word_idx, tag_idx, label_idx


if __name__ == '__main__':
    start = time.time()
    loader = Loader()
    train_x, train_y = loader.load_train_data()
    print('Done loading training data.')
    dev_sentences = loader.load_dev_data()
    print('Done loading dev data.')
    test_sentences = loader.load_test_data()
    print('Done loading test data.')
    print('Load data: {:.2f}s'.format(time.time() - start))


