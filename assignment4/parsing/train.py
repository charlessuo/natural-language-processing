import tensorflow as tf
import numpy as np
from loader import Loader


class MLP:
    def __init__(self, action_size, word_vocab_size, tag_vocab_size, label_vocab_size, hidden_size, batch_size=64, embed_size=50):
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.action_size = action_size

        self.train_word = tf.placeholder(tf.int32, shape=[self.batch_size], name='train_word')
        self.train_tag = tf.placeholder(tf.int32, shape=[self.batch_size], name='train_tag')
        self.train_label = tf.placeholder(tf.int32, shape=[self.batch_size], name='train_label')
        self.train_y = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='train_y')
        self.sess = tf.Session()

        with tf.name_scope('word_embedding'), tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform([word_vocab_size, self.embed_size], -0.01, 0.01), name='embeddings')
            embedded_word = tf.nn.embedding_lookup(embeddings, self.train_word, name='embedded_word')

        with tf.name_scope('tag_embedding'), tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform([tag_vocab_size, self.embed_size], -0.01, 0.01), name='embeddings')
            embedded_tag = tf.nn.embedding_lookup(embeddings, self.train_tag, name='embedded_tag')

        with tf.name_scope('label_embedding'), tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform([label_vocab_size, self.embed_size], -0.01, 0.01), name='embeddings')
            embedded_label = tf.nn.embedding_lookup(embeddings, self.train_label, name='embedded_label')

        self.inputs = tf.concat(1, [embedded_word, embedded_tag, embedded_label]) # shape=(batch_size, embed_size * 3)

        with tf.name_scope('hidden_layer'):
            W = tf.Variable(tf.truncated_normal([self.embed_size * 3, hidden_size],
                                                 stddev=1.0 / np.sqrt(self.embed_size * 3)), 
                                                 name='W')
            b = tf.Variable(tf.zeros([hidden_size]), name='b')
            self.h = (tf.matmul(self.inputs, W) + b) ** 3

        with tf.name_scope('output_layer'):
            W = tf.Variable(tf.truncated_normal([hidden_size, self.action_size], 
                                                 stddev=1.0 / np.sqrt(self.action_size)),
                                                 name='W')
            self.scores = tf.matmul(self.h, W)
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            one_hot_y = tf.one_hot(self.train_y, self.action_size)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.scores, one_hot_y), name='loss')

    def train(self, train_x, train_y, num_epoch=5):
        batch_pairs = self._extract_batch(train_x, train_y, num_epoch)
        optimizer = tf.train.AdamOptimizer(0.01)
        train_op = optimizer.minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

        for batch_x, batch_y in batch_pairs:
            batch_word = batch_x[:, 0]
            batch_tag = batch_x[:, 1]
            batch_label = batch_x[:, 2]
            _, loss = self.sess.run([train_op, self.loss], 
                                    feed_dict={self.train_word: batch_word, 
                                               self.train_tag: batch_tag, 
                                               self.train_label: batch_label, 
                                               self.train_y: batch_y})
            print(loss)


    def _extract_batch(self, x, y, num_epoch, shuffle=True):
        '''Extract batches from training data.
        Args:
            x: training data, containing 3 elements word, pos tag, and dep. labels 
                        as encoded indecies with shape (#samples, 3)
            y: training labels, actions in encoded form, shape (#samples,)
            num_epochs: int
            shuffle: bool
        Yield:
            batch_x, batch_y: Sliced data.
        '''
        # shuffle data
        data = np.hstack((x, y.reshape(-1, 1)))
        data_size = data.shape[0]
        batches_per_epoch = int((data_size - 1) / self.batch_size) + 1

        for epoch in range(num_epoch):
            if shuffle:
                shuffle_idxs = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_idxs]
            else:
                shuffled_data = data

            for batch_num in range(batches_per_epoch):
                start_idx = batch_num * self.batch_size
                end_idx = min((batch_num + 1) * self.batch_size, data_size)
                batch_x = shuffled_data[start_idx:end_idx][:, :x.shape[1]]
                batch_y = shuffled_data[start_idx:end_idx][:, x.shape[1]:]
                yield batch_x, batch_y

if __name__ == '__main__':
    loader = Loader()

    mlp = MLP(3, 10, 5, 3, 4, batch_size=2, embed_size=2)
    train_x = np.array([[1, 3, 0],
                        [0, 4, 2],
                        [7, 0, 1],
                        [6, 1, 2], 
                        [9, 1, 0], 
                        [8, 2, 1]])
    train_y = np.array([2, 0, 1, 0, 1, 2])
    mlp.train(train_x, train_y)


