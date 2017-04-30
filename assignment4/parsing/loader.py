import numpy as np
from collections import defaultdict, deque
import time

SHIFT = 'SHIFT'
LEFT_ARC = 'LEFT_ARC'
RIGHT_ARC = 'RIGHT_ARC'


class Loader:
    def __init__(self):
        self.data_path = './data/{}.conll'
        self.word_vocab = {}
        self.tag_vocab = {}
        self.label_vocab = {}

    def load_data(self, mode):
        sentences = self._load_raw_data(mode)
#        if mode == 'train':
#            self.word_vocab, self.tag_vocab, self.label_vocab = \
#                self._build_vocab(sentences)
        transitions = self._build_transitions(sentences[15:16])
        print(transitions)
    
    def _load_raw_data(self, mode):
        sentences = []
        with open(self.data_path.format(mode)) as f:
            sentence = []
            for line in f:
                line = line.split()
                if not line:
                    sentences.append(sentence)
                    sentence = []
                    continue
                id_, word, _, tag, tag, _, head, label, _, _ = line
                sentence.append([word, tag, label, id_, head])
        return sentences

    def _build_vocab(self, sentences):
        word_vocab = {'<UNK>': 0}
        tag_vocab = {'NULL': 0}
        label_vocab = {}

        word_counter = defaultdict(int)
        tag_counter = defaultdict(int)
        label_counter = defaultdict(int)
        for sentence in sentences:
            for item in sentence:
                word, tag, label, _, _ = item
                word_counter[word] += 1
                tag_counter[tag] += 1
                label_counter[label] += 1

        word_idx = 1 # leave space for <UNK>
        sorted_words = sorted(word_counter.items(), key=lambda x: (-x[1], x[0]))
        for word, _ in sorted_words:
            word_vocab[word] = word_idx
            word_idx += 1
            
        tag_idx = 1 # leave space for NULL
        sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], [0]))
        for tag, _ in sorted_tags:
            tag_vocab[tag] = tag_idx
            tag_idx += 1

        label_idx = 0
        sorted_labels = sorted(label_counter.items(), key=lambda x: (-x[1], [0]))
        for label, _ in sorted_labels:
            label_vocab[label] = label_idx
            label_idx += 1
        return word_vocab, tag_vocab, label_vocab

    def _build_sentence(self, sentences):
        for sentence in sentences:
            for item in sentence:
                word, tag, label = item
                
    def _build_transitions(self, sentences):
        transitions = []
        for sentence in sentences:
            transition = []
            stack = deque([['ROOT', None, 'ROOT', 0, None]]) # root
            buffer_ = deque(sentence)
            while buffer_:
                print('stack:')
                print([s[0] for s in stack])
                print('buffer:')
                print([b[0] for b in buffer_])
                s = stack.pop()
                s_id, s_head = s[-2:]
                b = buffer_.popleft()
                b_id, b_head = b[-2:]
                # LEFT_ARC
                if b_id == s_head:
                    transition.append((LEFT_ARC, s[2], s, b))
                    buffer_.appendleft(b)
                    print('LEFT_ARC')
                # RIGHT_ARC
                elif s_id == b_head:
                    transition.append((RIGHT_ARC, b[2], s, b))
                    buffer_.appendleft(s)
                    print('RIGHT_ARC')
                # SHIFT
                else:
                    transition.append((SHIFT, None, s, b))
                    stack.append(s)
                    stack.append(b)
                    print('SHIFT')
                print() 
            transitions.append(transition)
                
        return transitions

if __name__ == '__main__':
    start = time.time()
    loader = Loader()
    loader.load_data('train')
    print('Load data: {:.2f}s'.format(time.time() - start))


