#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from features import Loader


class NeuralNetwork:
    def __init__(self, vocab_size, num_classes, hidden_sizes):
        self.X = tf.placeholder(tf.float32, [None, vocab_size], name='X_train')
        self.Y = tf.placeholder(tf.float32, [None, num_classes], name='Y_train')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.h = [None] * len(hidden_sizes)
        self.sess = tf.Session()

        with tf.variable_scope('input'):
            W = tf.get_variable('W', [vocab_size, hidden_sizes[0]], initializer=self._xavier_weight_init())
            b = tf.get_variable('b', [hidden_sizes[0]], initializer=self._xavier_weight_init())
            self.h[0] = tf.nn.relu(tf.matmul(self.X, W) + b, name='relu')

        for i in range(1, len(hidden_sizes)):
            with tf.variable_scope('layer' + str(i)):
                W = tf.get_variable('W', [hidden_sizes[i - 1], hidden_sizes[i]], initializer=self._xavier_weight_init())
                b = tf.get_variable('b', [hidden_sizes[i]], initializer=self._xavier_weight_init())
                self.h[i] = tf.nn.relu(tf.matmul(self.h[i - 1], W) + b, name='relu')

        with tf.variable_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h[-1], self.dropout_keep_prob)

        with tf.variable_scope('output'):
            W = tf.get_variable('W', [hidden_sizes[-1], num_classes], initializer=self._xavier_weight_init())
            b = tf.get_variable('b', [num_classes], initializer=self._xavier_weight_init())
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.scores, self.Y))
        
        with tf.variable_scope('accuracy'):
            num_correct = tf.equal(self.predictions, tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(num_correct, 'float'), name='accuracy')

    def train(self, X, Y, num_epoch=20, batch_size=64):
        Y = self._onehot_encode(Y)
        batch_pairs = self._extract_batch(X, Y, num_epoch, batch_size)

        optimizer = tf.train.AdamOptimizer(1e-3)
        train_op = optimizer.minimize(self.loss)
        
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # start training
        batches_per_epoch = int((len(X) - 1) / batch_size) + 1
        step = 0
        for X_batch, Y_batch in batch_pairs:
            _, loss, accuracy = self.sess.run(
                [train_op, self.loss, self.accuracy], 
                feed_dict={self.X: X_batch, self.Y: Y_batch, self.dropout_keep_prob: 0.5})
            if step % batches_per_epoch == 0:
                print('Epoch: {}, loss: {:.6f}, accuracy: {:.2f}'.format(int(step / batches_per_epoch), loss, accuracy))
            step += 1
        return accuracy

    def predict(self, X_test):
        Y_pred = self.sess.run(self.predictions, feed_dict={self.X: X_test, self.dropout_keep_prob: 1.0})
        return Y_pred

    def score(self, X_dev, Y_dev):
        Y_dev = self._onehot_encode(Y_dev)
        loss, accuracy = self.sess.run(
            [self.loss, self.accuracy], feed_dict={self.X: X_dev, self.Y: Y_dev, self.dropout_keep_prob: 1.0})
        return accuracy

    def _onehot_encode(self, Y):
        '''Convert label vector Y into one-hot encoded form.'''
        Y_onehot = np.zeros((Y.shape[0], np.max(Y) + 1))
        for i in range(Y.shape[0]):
            Y_onehot[i][Y[i]] = 1
        return Y_onehot

    def _xavier_weight_init(self):
        '''
        Returns function that creates random tensor.
        The specified function will take in a shape (tuple or 1-d array)
        and return a random tensor of the specified shape drawn from 
        Xavier initialization distribution.
        '''
        def _xavier_initializer(shape, **kwargs):
            '''
            This function will be used as a variable scope initializer.
            Args:
                shape: Tuple or 1-d array that species dimensions of requested tensor.
            Returns:
                out: tf.Tensor of specified shape sampled from Xavier distribution.
            '''
            m = shape[0]
            n = shape[1] if len(shape) > 1 else shape[0]
            epsilon = np.sqrt(6) / np.sqrt(n + m)
            out = tf.random_uniform(shape, -epsilon, epsilon)
            return out
        return _xavier_initializer

    def _extract_batch(self, X, Y, num_epoch, batch_size, shuffle=True):
        '''Extract batches from training data.
        Args:
            X: training data, ndarray of one-hot encoding with shape (#samples, len of vocabulary)
            Y: training labels, labels in one-hot encoded form, shape (#samples, num_classes)
            num_epochs: int
            batch_size: int
            shuffle: bool
        Yield:
            X_batch, Y_batch: Sliced data.
        '''
        # shuffle data
        data = np.hstack((X, Y))
        data_size = data.shape[0]
        batches_per_epoch = int((data_size - 1) / batch_size) + 1

        for epoch in range(num_epoch):
            if shuffle:
                shuffle_idxs = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_idxs]
            else:
                shuffled_data = data

            for batch_num in range(batches_per_epoch):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, data_size)
                X_batch = shuffled_data[start_idx:end_idx][:, :X.shape[1]]
                Y_batch = shuffled_data[start_idx:end_idx][:, X.shape[1]:]
                yield X_batch, Y_batch


def generate_submission(Y_pred, class_dict, filename='submission'):
    with open('./results/' + filename + '.csv', 'w') as f:
        f.write('id,newsgroup\n')
        for i in range(len(Y_pred)):
            f.write('{},{}\n'.format(str(i), class_dict[Y_pred[i]]))


if __name__ == '__main__':
    loader = Loader('newsgroups')
    X_train, Y_train, X_dev, Y_dev, X_test = loader.tfidf(ngram_range=(1, 2), dim_used=20000)
    #X_train, Y_train, X_dev, Y_dev, X_test = loader.bow(ngram_range=(1, 2), dim_used=10000)
    print('Done loading data.')

    N, vocab_size = X_train.shape
    num_classes = np.max(Y_train) + 1

    nn = NeuralNetwork(vocab_size, num_classes, (256, 256))
    train_acc = nn.train(X_train, Y_train, num_epoch=50)
    print('Train Accuracy:', train_acc) # should overfit
    
    dev_acc = nn.score(X_dev, Y_dev)
    print('Dev Accuracy:', dev_acc)

#    Y_pred = nn.predict(X_test)
#    generate_submission(Y_pred, loader.class_dict, 'nn_256x2_n1to1_dim40000_tfidf_ep50')

