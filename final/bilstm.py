#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import logging
from loader import Loader

NUM_EPOCH      = 7
CELL_SIZE      = 64
EMBED_SIZE     = 64
POS_EMBED_SIZE = 32
BATCH_SIZE     = 32
NUM_LAYERS     = 1

FILENAME = 'bilstm_embed{0}_pos{1}_cell{2}_layer{3}_ep{4}'.format(
               EMBED_SIZE, POS_EMBED_SIZE, CELL_SIZE, NUM_LAYERS, NUM_EPOCH)
logging.basicConfig(filename='./logs/{}.log'.format(FILENAME), level=logging.DEBUG)


class RNN:
    def __init__(self, vocab_size, pos_vocab_size, num_classes, sentence_len, 
                 embed_size, pos_embed_size, cell_size, num_layers):
        self.sess = tf.Session()
        self.inputs = tf.placeholder(tf.int32, [None, sentence_len], name='inputs')
        self.pos_inputs = tf.placeholder(tf.int32, [None, sentence_len], name='pos_inputs')
        self.labels = tf.placeholder(tf.int32, [None, sentence_len], name='labels')
        self.seq_len = tf.reduce_sum(tf.cast(self.inputs > 0, tf.int32), axis=1)

        with tf.name_scope('word-embedding-layer'), tf.device('/cpu:0'):
            self.embeddings = tf.Variable(tf.random_uniform([vocab_size, embed_size], -0.1, 0.1), name='embeddings')
            self.embedded_x = tf.nn.embedding_lookup(self.embeddings, self.inputs, name='embedded_x')

        with tf.name_scope('pos-embedding-layer'), tf.device('/cpu:0'):
            self.embeddings = tf.Variable(tf.random_uniform([pos_vocab_size, pos_embed_size], -0.1, 0.1), name='embeddings')
            self.embedded_pos = tf.nn.embedding_lookup(self.embeddings, self.pos_inputs, name='embedded_pos')

        self.concat_input = tf.concat(2, [self.embedded_x, self.embedded_pos])
        
        with tf.name_scope('rnn-layer'):
            cell = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell,
                                                              cell,
                                                              dtype=tf.float32,
                                                              sequence_length=self.seq_len,
                                                              inputs=self.concat_input)
            self.rnn_output = tf.concat(2, outputs, name='rnn_output')

            # (N * sentence_len, 2 * cell_size)
            self.rnn_outputs_flat = tf.reshape(self.rnn_output, shape=[-1, 2 * cell_size])

        with tf.name_scope('output-layer'):
            W = tf.Variable(tf.truncated_normal([2 * cell_size, num_classes], 
                                                stddev=1.0 / np.sqrt(num_classes)), name='W')
            b = tf.Variable(tf.zeros([num_classes]), name='b')
            
            # (N * sentence_len, num_classes)
            self.logits_flat = tf.matmul(self.rnn_outputs_flat, W) + b
            probs_flat = tf.nn.softmax(self.logits_flat)
            self.probs = tf.reshape(probs_flat, [-1, sentence_len, num_classes])
            self.predictions = tf.argmax(self.probs, 2, name='predictions')
            self.predictions = tf.cast(self.predictions, tf.int32)

        with tf.name_scope('loss'):
            labels_flat = tf.reshape(self.labels, shape=[-1]) # (N * sentence_len,)
            unmasked_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits_flat, labels_flat)
            mask = tf.sign(tf.cast(labels_flat, tf.float32))
            masked_loss = tf.reshape(unmasked_loss * mask, shape=[-1, sentence_len])
            self.loss = tf.reduce_mean(tf.reduce_sum(masked_loss, axis=1) / tf.cast(self.seq_len, tf.float32), name='loss')
        
    def train(self, train_x, train_x_pos, train_y, id_to_class, num_epoch=3, batch_size=64, tune_ratio=None):
        # set optimizer
        optimizer = tf.train.AdamOptimizer(0.01)

        # clipping gradient
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 1)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.sess.run(tf.global_variables_initializer())

        # prepare batches to train on
        batch_pairs = self._extract_batch(train_x, train_x_pos, train_y, num_epoch, batch_size, tune_ratio, shuffle=True)
        batches_per_epoch = int((len(train_x) - 1) / batch_size) + 1
        step = 0
        tune_info = []

        # start training 
        for batch in batch_pairs:
            batch_x, batch_x_pos, batch_y, tune_x, tune_y = batch
            _, loss = self.sess.run([train_op, self.loss], 
                                    feed_dict={self.inputs: batch_x, self.pos_inputs: batch_x_pos, self.labels: batch_y})
            if step % 10 == 0:
                batch_f1, _, _ = self.calculate_f1(batch_x, batch_x_pos, batch_y, id_to_class)
                print('[Batch]: Epoch: {}, Step: {}, loss: {:.6f}, F1: {:.4f}'.format(
                      int(step / batches_per_epoch), step, loss, batch_f1))
                logging.debug('[Batch]: Epoch: {}, Step: {}, loss: {:.6f}, F1: {:.4f}'.format(
                              int(step / batches_per_epoch), step, loss, batch_f1))

            if step % 100 == 0 and tune_x is not None and tune_y is not None:
                tune_loss = self.sess.run(self.loss, feed_dict={self.inputs: tune_x, self.labels: tune_y})
                tune_f1, _, _ = self.calculate_f1(tune_x, tune_y, id_to_class)
                print('[Tune]: Epoch: {}, Step: {}, loss: {:.6f}, F1: {:.4f}'.format(
                      int(step / batches_per_epoch), step, tune_loss, tune_f1))
                logging.debug('[Tune]: Epoch: {}, Step: {}, loss: {:.6f}, F1: {:.4f}'.format(
                              int(step / batches_per_epoch), step, tune_loss, tune_f1))
                tune_info.append(tune_loss)
            step += 1

    def _extract_batch(self, train_x, train_x_pos, train_y, num_epoch, batch_size, tune_ratio, shuffle=True):
        data = np.hstack((train_x, train_x_pos, train_y))
        if tune_ratio:
            tune_size = int(data.shape[0] * tune_ratio)
        else:
            tune_size = 0
        train_size = data.shape[0] - tune_size
        batches_per_epoch = int((train_size - 1) / batch_size) + 1

        for epoch in range(num_epoch):
            # extract tuning set
            tune_idxs = np.random.choice(len(data), tune_size, replace=False)
            if tune_ratio:
                tune_x = data[tune_idxs][:, :train_x.shape[1]]
                tune_y = data[tune_idxs][:, train_x.shape[1]:]
            else:
                tune_x = None
                tune_y = None

            # shuffle and build batches with the reset
            data_ = np.delete(data, tune_idxs, axis=0)
            if shuffle:
                shuffle_idxs = np.random.permutation(np.arange(train_size))
                shuffled_data = data_[shuffle_idxs]
            else:
                shuffled_data = data_

            for batch_num in range(batches_per_epoch):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, train_size)
                batch_x = shuffled_data[start_idx:end_idx][:, :train_x.shape[1]]
                batch_x_pos = shuffled_data[start_idx:end_idx][:, train_x.shape[1]:train_x.shape[1] + train_x_pos.shape[1]]
                batch_y = shuffled_data[start_idx:end_idx][:, train_x.shape[1] + train_x_pos.shape[1]:]
                yield batch_x, batch_x_pos, batch_y, tune_x, tune_y

    def predict(self, x, x_pos):
        preds = self.sess.run(self.predictions, feed_dict={self.inputs: x, self.pos_inputs: x_pos})
        return preds

    def calculate_f1(self, x, x_pos, y, id_to_class):
        predictions, seq_len = self.sess.run([self.predictions, self.seq_len], 
                                             feed_dict={self.inputs: x, self.pos_inputs: x_pos})
        true_pos = 0
        false_pos = 0
        false_neg = 0
        for i in range(len(predictions)):
            for t in range(seq_len[i]):
                y_ = id_to_class[y[i, t]]
                p = id_to_class[predictions[i, t]]
                if y_ != 'O':
                    if p == y_:
                        true_pos += 1
                    elif p == 'O':
                        false_neg += 1
                elif p != y_:
                    false_pos += 1
        
        prec = true_pos / (true_pos + false_pos) if true_pos + false_pos != 0 else 0
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg != 0 else 0
        f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg) if 2 * true_pos + false_pos + false_neg != 0 else 0
        return f1, prec, recall

    def generate_submission(self, test_preds, test_x, id_to_class, filename='submission'):
        test_seq_len = self.sess.run(self.seq_len, feed_dict={self.inputs: test_x})
        with open('./results/' + filename, 'w') as f:
            idx = 0
            for i in range(len(test_preds)):
                for j in range(test_seq_len[i]):
                    f.write('{}\n'.format(id_to_class[test_preds[i, j]]))
                    idx += 1
                f.write('\n')


if __name__ == '__main__':
    loader = Loader()
    train_x, train_x_pos, train_y = loader.load_data('train')
    dev_x, dev_x_pos, dev_y = loader.load_data('dev')
    test_x, test_x_pos, _ = loader.load_data('test')

    id_to_word = loader.id_to_word
    id_to_class = loader.id_to_class
    id_to_pos = loader.id_to_pos
    max_len = loader.max_len

    rnn = RNN(vocab_size=len(id_to_word), 
              pos_vocab_size=len(id_to_pos),
              num_classes=len(id_to_class), 
              sentence_len=max_len, 
              embed_size=EMBED_SIZE, 
              pos_embed_size=POS_EMBED_SIZE,
              cell_size=CELL_SIZE,
              num_layers=NUM_LAYERS)
    rnn.train(train_x, train_x_pos, train_y, id_to_class, 
              num_epoch=NUM_EPOCH, batch_size=BATCH_SIZE, tune_ratio=None)
    dev_f1, dev_prec, dev_rec = \
        rnn.calculate_f1(dev_x, dev_x_pos, dev_y, id_to_class)

    print('[Dev]: F1 = {:.4f}, Prec = {:.4f}, Recall = {:.4f}'
          .format(dev_f1, dev_prec, dev_rec))
    logging.debug('[Dev]: F1 = {:.4f}, Prec = {:.4f}, Recall = {:.4f}'
                  .format(dev_f1, dev_prec, dev_rec))
    
#    train_preds = rnn.predict(train_x, train_x_pos)
#    rnn.generate_submission(train_preds, train_x, id_to_class, 
#                            filename=(FILENAME + '.train'))

    dev_preds = rnn.predict(dev_x, dev_x_pos)
    rnn.generate_submission(dev_preds, dev_x, id_to_class, 
                            filename=(FILENAME + '.dev'))

#    test_preds = rnn.predict(test_x, test_x_pos)
#    rnn.generate_submission(test_preds, test_x, id_to_class, 
#                            filename=(FILENAME + '.test'))



