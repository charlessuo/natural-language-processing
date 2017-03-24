#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from loader import Loader


class RNN:
    def __init__(self, vocab_size, num_classes, sentence_len, embed_size=300, cell_size=128, num_layers=1):
        self.sess = tf.Session()
        self.inputs = tf.placeholder(tf.int32, [None, sentence_len], name='inputs')
        self.labels = tf.placeholder(tf.int32, [None, sentence_len], name='labels')
        self.seq_len = tf.reduce_sum(tf.cast(self.labels > 0, tf.int32), axis=1)

        with tf.name_scope('embedding-layer'), tf.device('/cpu:0'):
            self.embeddings = tf.Variable(tf.random_uniform([vocab_size, embed_size], -1.0, 1.0), name='embeddings')
            self.embedded_x = tf.nn.embedding_lookup(self.embeddings, self.inputs, name='embedded_x')
        
        with tf.name_scope('rnn-layer'):
            cell = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell,
                                                              cell,
                                                              dtype=tf.float32,
                                                              sequence_length=self.seq_len,
                                                              inputs=self.embedded_x)
            self.rnn_output = tf.concat(2, outputs, name='rnn_output')
            self.rnn_outputs_flat = tf.reshape(self.rnn_output, shape=[-1, 2 * cell_size])
        
        with tf.name_scope('output-layer'):
            W = tf.Variable(tf.truncated_normal([2 * cell_size, num_classes], stddev=1.0 / np.sqrt(num_classes)), name='W')
            b = tf.Variable(tf.zeros([num_classes]), name='b')

            # rnn_outputs_flat: shape=(N * sentence_len, 2 * cell_size) 
            # logits_flat:      shape=(N * sentence_len, num_classes)
            self.logits_flat = tf.matmul(self.rnn_outputs_flat, W) + b
            probs_flat = tf.nn.softmax(self.logits_flat)

            labels_flat = tf.reshape(self.labels, shape=[-1]) # shape=(N * sentence_len,)
            self.scores = tf.reshape(probs_flat, [-1, sentence_len, num_classes])
            self.predictions = tf.argmax(self.scores, 2, name='predictions')
            pred_mask = tf.reshape(tf.sign(labels_flat), shape=[-1, sentence_len]) # shape=(N, sentence_len)
            self.predictions = tf.cast(self.predictions, tf.int32)
            self.predictions *= pred_mask
        
        with tf.name_scope('loss'):
            unmasked_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits_flat, labels_flat)
            mask = tf.sign(tf.cast(labels_flat, tf.float32))
            masked_loss = tf.reshape(unmasked_loss * mask, shape=[-1, sentence_len])
            self.loss = tf.reduce_mean(tf.reduce_sum(masked_loss, axis=1) / tf.cast(self.seq_len, tf.float32), name='loss')
        
        with tf.name_scope('accuracy'):
            num_correct = tf.equal(self.predictions, self.labels) #TODO: need to fix this
            self.accuracy = tf.reduce_mean(tf.cast(num_correct, 'float'), name='accuracy')
        
    def train(self, train_x, train_y, num_epoch=20, batch_size=32):
        batch_pairs = self._extract_batch(train_x, train_y, num_epoch, batch_size, shuffle=True)

        optimizer = tf.train.AdamOptimizer(0.1)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 1)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.sess.run(tf.global_variables_initializer())

        batches_per_epoch = int((len(train_x) - 1) / batch_size) + 1
        step = 0
        for batch_x, batch_y in batch_pairs:
            _, loss, acc = self.sess.run([train_op, self.loss, self.accuracy], 
                                           feed_dict={self.inputs: batch_x, self.labels: batch_y})
            if step % 10 == 0:
                print('Epoch: {}, Step: {}, loss: {:.6f}, accuracy: {:.2f}'.format(int(step / batches_per_epoch), 
                                                                                   step,
                                                                                   loss, 
                                                                                   acc))
                print('Prediction for the first row in batch:')
                print(self.sess.run(self.predictions, feed_dict={self.inputs: batch_x[:1, :], self.labels: batch_y[:1, :]}))
                print('Label for the first row in batch:')
                print(self.sess.run(self.labels, feed_dict={self.labels: batch_y[:1, :]}))
            step += 1

    def predict(self, test_x):
        pred = self.sess.run(self.predictions, feed_dict={self.inputs: test_x})
        return pred

    def score(self, dev_x, dev_y):
        accuracy = self.sess.run(self.accuracy, feed_dict={self.inputs: dev_x, self.labels: dev_y})
        return accuracy

    def _extract_batch(self, train_x, train_y, num_epoch, batch_size, shuffle=True):
        data = np.hstack((train_x, train_y))
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
                batch_x = shuffled_data[start_idx:end_idx][:, :train_x.shape[1]]
                batch_y = shuffled_data[start_idx:end_idx][:, train_x.shape[1]:]
                yield batch_x, batch_y


if __name__ == '__main__':
    loader = Loader()
    train_x, train_y = loader.load('train')
    dev_x, dev_y = loader.load('dev')

    id_to_word = loader.id_to_word
    id_to_class = loader.id_to_class
    max_len = loader.max_len

    rnn = RNN(len(id_to_word), len(id_to_class), max_len, embed_size=100, cell_size=128, num_layers=2)
    rnn.train(train_x, train_y, num_epoch=10, batch_size=64)

