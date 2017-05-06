import tensorflow as tf
import numpy as np
from collections import deque
import time
from loader import Loader


class MLP:
    def __init__(self, action_size, word_vocab_size, tag_vocab_size, label_vocab_size, 
                       num_w=2, num_t=2, num_l=2, hidden_size=None, embed_size=50, batch_size=64):
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.action_size = action_size
        self.num_w = num_w
        self.num_t = num_t
        self.num_l = num_l

        self.train_word = tf.placeholder(tf.int32, shape=[None, self.num_w])
        self.train_tag = tf.placeholder(tf.int32, shape=[None, self.num_t])
        self.train_label = tf.placeholder(tf.int32, shape=[None, self.num_l])
        self.train_y = tf.placeholder(tf.int32, shape=[None])
        self.sess = tf.Session()

        with tf.name_scope('word_embedding'), tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform(
                             [word_vocab_size, self.embed_size], -0.1, 0.1))
            embedded_word_stacked = tf.nn.embedding_lookup(embeddings, self.train_word)
            word_stack = []
            for i in range(self.num_w):
                word_stack.append(tf.slice(embedded_word_stacked, [0, i, 0], 
                                           [-1, 1, self.embed_size]))
            embedded_word = tf.reshape(tf.concat(2, word_stack), 
                                       [-1, self.embed_size * self.num_w])

        with tf.name_scope('tag_embedding'), tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform(
                             [tag_vocab_size, self.embed_size], -0.1, 0.1))
            embedded_tag_stacked = tf.nn.embedding_lookup(embeddings, self.train_tag)
            tag_stack = []
            for i in range(self.num_t):
                tag_stack.append(tf.slice(embedded_tag_stacked, [0, i, 0], 
                                          [-1, 1, self.embed_size]))
            embedded_tag = tf.reshape(tf.concat(2, tag_stack), 
                                      [-1, self.embed_size * self.num_t])

        with tf.name_scope('label_embedding'), tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform(
                             [label_vocab_size, self.embed_size], -0.1, 0.1))
            embedded_label_stacked = tf.nn.embedding_lookup(embeddings, self.train_label)
            label_stack = []
            for i in range(self.num_l):
                label_stack.append(tf.slice(embedded_label_stacked, [0, i, 0], 
                                            [-1, 1, self.embed_size]))
            embedded_label = tf.reshape(tf.concat(2, label_stack), 
                                        [-1, self.embed_size * self.num_l])

        self.inputs = tf.concat(1, [embedded_word, embedded_tag, embedded_label])

        with tf.name_scope('hidden_layer'):
            input_dim = self.embed_size * (self.num_w + self.num_t + self.num_l)
            W = tf.Variable(tf.truncated_normal([input_dim, hidden_size],
                                                 stddev=1.0 / np.sqrt(input_dim)),
                                                 name='W')
            b = tf.Variable(tf.zeros([hidden_size]), name='b')
            self.h = tf.sigmoid(tf.matmul(self.inputs, W) + b)

        with tf.name_scope('output_layer'):
            W = tf.Variable(tf.truncated_normal([hidden_size, self.action_size], 
                                                 stddev=1.0 / np.sqrt(self.action_size)),
                                                 name='W')
            self.scores = tf.matmul(self.h, W)
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            one_hot_y = tf.one_hot(self.train_y, self.action_size)
            self.loss = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(self.scores, one_hot_y))

        with tf.variable_scope('accuracy'):
            num_correct = tf.equal(self.predictions, tf.argmax(one_hot_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(num_correct, tf.float32))

    def train(self, train_x, train_y, num_epoch=3):
        batch_pairs = self._extract_batch(train_x, train_y, num_epoch)
        optimizer = tf.train.AdagradOptimizer(0.01)
        train_op = optimizer.minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

        batches_per_epoch = int((len(train_x) - 1) / self.batch_size) + 1
        step = 0
        for batch_x, batch_y in batch_pairs:
            batch_word = batch_x[:, :self.num_w]
            batch_tag = batch_x[:, self.num_w:self.num_w + self.num_t]
            batch_label = batch_x[:, self.num_w + self.num_t:]
            _, loss, acc = self.sess.run([train_op, self.loss, self.accuracy], 
                                         feed_dict={self.train_word: batch_word, 
                                                    self.train_tag: batch_tag, 
                                                    self.train_label: batch_label, 
                                                    self.train_y: batch_y})
            if step % 100 == 0:
                print('[Train]: Epoch: {}, Step: {}'
                      ', loss: {:.6f}, accuracy: {:.4f}'.format(
                      int(step / batches_per_epoch), step, loss, acc))
            step += 1

    def _extract_batch(self, x, y, num_epoch, shuffle=True):
        '''Extract batches from training data.
        Args:
            x: training data, containing elements of words, pos tags, and dep. labels 
                        as encoded indecies with shape (#samples, num_w + num_t + 1)
            y: training labels, actions in encoded form, shape (#samples,)
            num_epoch: int
            shuffle: bool
        Yield:
            batch_x, batch_y: Sliced data.
        '''
        # shuffle data
        data_size = x.shape[0]
        batches_per_epoch = int((data_size - 1) / self.batch_size) + 1

        for epoch in range(num_epoch):
            if shuffle:
                shuffle_idxs = np.random.permutation(np.arange(data_size))
                shuffled_x = x[shuffle_idxs]
                shuffled_y = y[shuffle_idxs]
            else:
                shuffled_x = x
                shuffled_y = y

            for batch_num in range(batches_per_epoch):
                start_idx = batch_num * self.batch_size
                end_idx = min((batch_num + 1) * self.batch_size, data_size)
                batch_x = shuffled_x[start_idx:end_idx]
                batch_y = shuffled_y[start_idx:end_idx]
                yield batch_x, batch_y

    def parse(self, sentences, word_vocab, tag_vocab, label_vocab, rev_action_space):
        results = []
        for i, sentence in enumerate(sentences):
            result = []
            stack = deque([['<ROOT>', 'NULL', 'ROOT', 0, None]])
            buffer_ = deque(sentence)
            while buffer_:
                if not stack:
                    # see empty stack, do SHIFT and go to the next round
                    b = buffer_.popleft()
                    stack.append(b)
                    continue
                s = stack.pop()
                s_word, s_tag, s_label, s_id, _ = s
                s_word_idx, s_tag_idx, s_label_idx = \
                    self._map_to_idx(s, word_vocab, tag_vocab, label_vocab)
                b = buffer_.popleft()
                b_word, b_tag, b_label, b_id, _ = b
                b_word_idx, b_tag_idx, b_label_idx = \
                    self._map_to_idx(b, word_vocab, tag_vocab, label_vocab)
                action_idx = self.sess.run(self.predictions, feed_dict={
                                 self.train_word: [[s_word_idx, b_word_idx]], 
                                 self.train_tag: [[s_tag_idx, b_tag_idx]], 
                                 self.train_label: [[s_label_idx, b_label_idx]]})
                action, label = rev_action_space[int(action_idx)]

                if action == 'LEFT_ARC' and s_word != '<ROOT>':
                    buffer_.appendleft(b)
                    result.append([int(s_id), s_word, s_tag, b_id, label])
                elif action == 'RIGHT_ARC':
                    buffer_.appendleft(s)
                    result.append([int(b_id), b_word, b_tag, s_id, label])
                else:
                    # 'SHIFT'
                    stack.append(s)
                    stack.append(b)

            # buffer ended up empty but still got items in stack
            # run RIGHT_ARC utill there's only '<ROOT>' left
            while len(stack) > 1:
                s_word, s_tag, s_label, s_id, _ = stack.pop()
                head_id = stack[-1][3]
                result.append([int(s_id), s_word, s_tag, head_id, s_label])
            results.append(sorted(result))

            if i % 500 == 0:
                print('[Parse]: {} sentences parsed.'.format(i))
        return results

    def _map_to_idx(self, node, word_vocab, tag_vocab, label_vocab):
        word, tag, label, _, _ = node
        if word in word_vocab:
            word_idx = word_vocab[word]
        else:
            word_idx = 0
        tag_idx = tag_vocab[tag]
        label_idx = label_vocab[label]
        return word_idx, tag_idx, label_idx

def write_to_conll(results, filename='result.conll'):
    with open('./results/' + filename, 'w') as f:
        for result in results:
            for item in result:
                id_, word, tag, head, label = item
                f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n'.format(
                         id_, word, '_', tag, tag, '_', head, label, '_', '_'))
            f.write('\n')


if __name__ == '__main__':
    print('Start training.')
    start = time.time()
    loader = Loader()
    train_x, train_y = loader.load_train_data()
    dev_sentences = loader.load_dev_data()
    print('Done loading in {:.2f} min'.format((time.time() - start) / 60))
    print('Dev number of sentences:', len(dev_sentences))
#    test_sentences = loader.load_test_data()
    
    word_vocab_size = len(loader.word_vocab)
    tag_vocab_size = len(loader.tag_vocab)
    label_vocab_size = len(loader.label_vocab)
    action_size = np.max(train_y) + 1

    word_vocab = loader.word_vocab
    tag_vocab = loader.tag_vocab
    label_vocab = loader.label_vocab
    rev_action_space = loader.rev_action_space

    mlp = MLP(action_size, word_vocab_size, tag_vocab_size, label_vocab_size,
              num_w=2, num_t=2, hidden_size=300, embed_size=50, batch_size=64)

    print('Start training.')
    start = time.time()
    mlp.train(train_x, train_y, num_epoch=1)
    print('Done training in {:.2f} min'.format((time.time() - start) / 60))

    print('Start parsing using MLP.')
    start = time.time()
    results = mlp.parse(dev_sentences[:3000], 
                        word_vocab, tag_vocab, label_vocab, rev_action_space)
    print('Done parsing in {:.2f} min'.format((time.time() - start) / 60))
    
    write_to_conll(results)

