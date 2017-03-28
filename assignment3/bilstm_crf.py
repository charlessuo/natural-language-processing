#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import logging
from rnn_loader import Loader

FILENAME = 'sample-crf'
logging.basicConfig(filename='./log/{}.log'.format(FILENAME), level=logging.DEBUG)


class RNN:
    def __init__(self, vocab_size, num_classes, sentence_len, embed_size, cell_size, num_layers):
        self.sess = tf.Session()
        self.inputs = tf.placeholder(tf.int32, [None, sentence_len], name='inputs')
        self.labels = tf.placeholder(tf.int32, [None, sentence_len], name='labels')
        self.seq_len = tf.reduce_sum(tf.cast(self.inputs > 0, tf.int32), axis=1)

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
            self.rnn_outputs_flat = tf.reshape(self.rnn_output, shape=[-1, 2 * cell_size]) # (N * sentence_len, 2 * cell_size)

        with tf.name_scope('output-layer'):
            W = tf.Variable(tf.truncated_normal([2 * cell_size, num_classes], stddev=1.0 / np.sqrt(num_classes)), name='W')
            b = tf.Variable(tf.zeros([num_classes]), name='b')

            self.logits_flat = tf.matmul(self.rnn_outputs_flat, W) + b # (N * sentence_len, num_classes)

            self.logits = tf.reshape(self.logits_flat, [-1, sentence_len, num_classes])
            log_likelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.seq_len)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(-log_likelihood)
        
    def train(self, train_x, train_y, num_epoch=3, batch_size=64):
        batch_pairs = self._extract_batch(train_x, train_y, num_epoch, batch_size, shuffle=True)

        optimizer = tf.train.AdamOptimizer(0.01)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 1)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.sess.run(tf.global_variables_initializer())

        batches_per_epoch = int((len(train_x) - 1) / batch_size) + 1
        step = 0
        for batch_x, batch_y in batch_pairs:
            _, loss = self.sess.run([train_op, self.loss], 
                                    feed_dict={self.inputs: batch_x, self.labels: batch_y})
            acc = self.calculate_accuracy(batch_x, batch_y)
            if step % 10 == 0:
                print('Epoch: {}, Step: {}, loss: {:.6f}, accuracy: {:.2f}'.format(
                      int(step / batches_per_epoch), step, loss, acc))
                logging.debug('Epoch: {}, Step: {}, loss: {:.6f}, accuracy: {:.2f}'.format(
                              int(step / batches_per_epoch), step, loss, acc))

                print('Prediction for the first row in batch:')
                print(self.predict(batch_x[:1, :]))
                print('Label for the first row in batch:')
                print(self.sess.run(self.labels, feed_dict={self.labels: batch_y[:1, :]}))
            step += 1

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

    def predict(self, x):
        logits, seq_len, trans_params = self.sess.run([self.logits, self.seq_len, self.trans_params],
                                                      feed_dict={self.inputs: x})
        N, sentence_len, _ = logits.shape
        preds = np.zeros((N, sentence_len))
        for i, logit_, length in zip(range(N), logits, seq_len):
            # remove padding from the score and tag sequences
            logit_ = logit_[:length]
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit_, trans_params)
            preds[i, :length] = np.array(viterbi_seq)
        return preds.astype(int)

    def calculate_accuracy(self, x, y):
        logits, labels, seq_len, trans_params = self.sess.run([self.logits, self.labels, self.seq_len, self.trans_params],
                                                              feed_dict={self.inputs: x, self.labels: y})

        num_correct = 0
        for logit_, y_, length in zip(logits, labels, seq_len):
            # remove padding from the score and tag sequences
            logit_ = logit_[:length]
            y_ = y_[:length]
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit_, trans_params)
            num_correct += np.sum(np.equal(viterbi_seq, y_))
        return num_correct / np.sum(seq_len)

    def generate_submission(self, test_preds, test_x, id_to_class, filename='submission'):
        test_seq_len = self.sess.run(self.seq_len, feed_dict={self.inputs: test_x})
        with open('./results/' + filename + '.csv', 'w') as f:
            f.write('id,tag\n')
            idx = 0
            for i in range(len(test_preds)):
                for j in range(test_seq_len[i]):
                    f.write('{},"{}"\n'.format(str(idx), id_to_class[test_preds[i, j]]))
                    idx += 1


if __name__ == '__main__':
    loader = Loader()
    train_x, train_y = loader.load_data('train')
    dev_x, dev_y = loader.load_data('dev')
    test_x, _ = loader.load_data('test')

    id_to_word = loader.id_to_word
    id_to_class = loader.id_to_class
    max_len = loader.max_len

    rnn = RNN(vocab_size=len(id_to_word), 
              num_classes=len(id_to_class), 
              sentence_len=max_len, 
              embed_size=64, 
              cell_size=64, 
              num_layers=1)
    rnn.train(train_x[:3200], train_y[:3200], num_epoch=1, batch_size=64)
    dev_accuracy = rnn.calculate_accuracy(dev_x, dev_y)
    print('Dev accuracy:', dev_accuracy)
    logging.debug('Dev accuracy: {}'.format(dev_accuracy))

    test_preds = rnn.predict(test_x)
    rnn.generate_submission(test_preds, test_x, id_to_class, filename=FILENAME)

