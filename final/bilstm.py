#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import logging
from loader import Loader
from orth_loader import OrthLoader


NUM_EPOCH       = 7
EMBED_SIZE      = 64
POS_EMBED_SIZE  = 32
ORTH_EMBED_SIZE = 32
CELL_SIZE       = 64
BATCH_SIZE      = 32
NUM_LAYERS      = 1

FILENAME = 'test'
#FILENAME = 'bilstm_embed{0}_pos{1}_cell{2}_layer{3}_ep{4}'.format(
#               EMBED_SIZE, POS_EMBED_SIZE, CELL_SIZE, NUM_LAYERS, NUM_EPOCH)
logging.basicConfig(filename='./logs/{}.log'.format(FILENAME), level=logging.DEBUG)


class BiLSTM:
    def __init__(self, vocab_size, pos_vocab_size, orth_vocab_size, num_classes, sentence_len, 
                 embed_size, pos_embed_size, orth_embed_size, cell_size, num_layers):

        self.sess = tf.Session()
        self.inputs = tf.placeholder(tf.int32, [None, sentence_len], name='inputs')
        self.pos_inputs = tf.placeholder(tf.int32, [None, sentence_len], name='pos_inputs')
        self.orth_inputs = tf.placeholder(tf.int32, [None, sentence_len], name='orth_inputs')
        self.labels = tf.placeholder(tf.int32, [None, sentence_len], name='labels')
        self.seq_len = tf.reduce_sum(tf.cast(self.inputs > 0, tf.int32), axis=1)
        l2_loss = tf.constant(0.0)

        with tf.name_scope('word-embedding-layer'), tf.device('/cpu:0'):
            self.embeddings = tf.Variable(tf.random_uniform([vocab_size, embed_size], -0.1, 0.1), name='embeddings')
            self.embedded_x = tf.nn.embedding_lookup(self.embeddings, self.inputs, name='embedded_x')

        with tf.name_scope('pos-embedding-layer'), tf.device('/cpu:0'):
            self.embeddings = tf.Variable(tf.random_uniform([pos_vocab_size, pos_embed_size], -0.1, 0.1), name='embeddings')
            self.embedded_pos = tf.nn.embedding_lookup(self.embeddings, self.pos_inputs, name='embedded_pos')

        with tf.name_scope('orthographic-embedding-layer'), tf.device('/cpu:0'):
            self.embeddings = tf.Variable(tf.random_uniform([orth_vocab_size, orth_embed_size], -0.1, 0.1), name='embeddings')
            self.embedded_orth = tf.nn.embedding_lookup(self.embeddings, self.orth_inputs, name='embedded_orth')

        self.concat_input = tf.concat(2, [self.embedded_x, self.embedded_pos, self.embedded_orth])
        
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
        '''
        with tf.name_scope('output-layer'):
            W = tf.Variable(tf.truncated_normal([2 * cell_size, num_classes], 
                                                stddev=1.0 / np.sqrt(num_classes)), name='W')
            b = tf.Variable(tf.zeros([num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

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
#            self.loss += 0.01 * l2_loss
        '''
        with tf.name_scope('output-layer'):
            W = tf.Variable(tf.truncated_normal([2 * cell_size, num_classes],
                                                stddev=1.0 / np.sqrt(num_classes)), name='W')
            b = tf.Variable(tf.zeros([num_classes]), name='b')

            # (N * sentence_len, num_classes)
            self.logits_flat = tf.matmul(self.rnn_outputs_flat, W) + b

            self.logits = tf.reshape(self.logits_flat, [-1, sentence_len, num_classes])
            log_likelihood, self.trans_params = \
                tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.seq_len)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(-log_likelihood)
        
    def train(self, train_x, train_pos, train_orth, train_y, id_to_class, num_epoch=3, batch_size=64):
        # set optimizer
        optimizer = tf.train.AdamOptimizer(0.01)

        # clipping gradient
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 1)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.sess.run(tf.global_variables_initializer())

        # prepare batches to train on
        batch_pairs = self._extract_batch(train_x, train_pos, train_orth, train_y, num_epoch, batch_size)
        batches_per_epoch = int((len(train_x) - 1) / batch_size) + 1
        step = 0

        # start training 
        for batch in batch_pairs:
            batch_x, batch_pos, batch_orth, batch_y = batch
            _, loss = self.sess.run([train_op, self.loss], 
                                    feed_dict={self.inputs: batch_x, 
                                               self.pos_inputs: batch_pos, 
                                               self.orth_inputs: batch_orth, 
                                               self.labels: batch_y})
            if step % 10 == 0:
                batch_f1, _, _ = self.calculate_f1(batch_x, batch_pos, batch_orth, batch_y, id_to_class)
                print('[Batch]: Epoch: {}, Step: {}, loss: {:.6f}, F1: {:.4f}'.format(
                      int(step / batches_per_epoch), step, loss, batch_f1))
                logging.debug('[Batch]: Epoch: {}, Step: {}, loss: {:.6f}, F1: {:.4f}'.format(
                              int(step / batches_per_epoch), step, loss, batch_f1))
            step += 1

    def _extract_batch(self, train_x, train_pos, train_orth, train_y, num_epoch, batch_size):
        train_size = train_x.shape[0]
        batches_per_epoch = int((train_size - 1) / batch_size) + 1

        for epoch in range(num_epoch):
            # shuffle and build batches with the reset
            shuffle_idxs = np.random.permutation(np.arange(train_size))

            for batch_num in range(batches_per_epoch):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, train_size)
                batch_x = train_x[start_idx:end_idx]
                batch_pos = train_pos[start_idx:end_idx]
                batch_orth = train_orth[start_idx:end_idx]
                batch_y = train_y[start_idx:end_idx]
                yield batch_x, batch_pos, batch_orth, batch_y

    def cnn_char_embedding(self, sentence):
        pass

    def predict(self, x, pos, orth):
#        preds = self.sess.run(self.predictions, feed_dict={self.inputs: x, 
#                                                           self.pos_inputs: pos,
#                                                           self.orth_inputs: orth})
        logits, seq_len, trans_params = self.sess.run([self.logits, self.seq_len, self.trans_params],
                                                      feed_dict={self.inputs: x, self.pos_inputs: pos, self.orth_inputs: orth})
        N, sentence_len, _ = logits.shape
        preds = np.zeros((N, sentence_len))
        for i, logit_, length in zip(range(N), logits, seq_len):
            # remove padding from the score and tag sequences
            logit_ = logit_[:length]
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit_, trans_params)
            preds[i, :length] = np.array(viterbi_seq)
        preds = preds.astype(int)
        return preds

    def calculate_f1(self, x, pos, orth, y, id_to_class):
        logits, labels, seq_len, trans_params = self.sess.run([self.logits, self.labels, self.seq_len, self.trans_params],
                                                              feed_dict={self.inputs: x, 
                                                                         self.pos_inputs: pos, 
                                                                         self.orth_inputs: orth, 
                                                                         self.labels: y})
        true_pos = 0
        false_pos = 0
        false_neg = 0        
        for logit_, y_seq, length in zip(logits, labels, seq_len):
            # remove padding from the score and tag sequences
            logit_ = logit_[:length]
            y_seq = y_seq[:length]
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit_, trans_params)
            for y_id, p_id in zip(y_seq, viterbi_seq):
                y_ = id_to_class[y_id]
                p = id_to_class[p_id]
                if y_ != 'O':
                    if p == y_:
                        true_pos += 1
                    elif p == 'O':
                        false_neg += 1
                elif p != y_:
                    false_pos += 1
        prec = true_pos / (true_pos + false_pos) if true_pos + false_pos != 0 else 0
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg != 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall != 0 else 0
        return f1, prec, recall            

#        predictions, seq_len = self.sess.run([self.predictions, self.seq_len], 
#                                             feed_dict={self.inputs: x, 
#                                                        self.pos_inputs: pos,
#                                                        self.orth_inputs: orth})
#
#        true_pos = 0
#        false_pos = 0
#        false_neg = 0
#        for i in range(len(predictions)):
#            for t in range(seq_len[i]):
#                y_ = id_to_class[y[i, t]]
#                p = id_to_class[predictions[i, t]]
#                if y_ != 'O':
#                    if p == y_:
#                        true_pos += 1
#                    elif p == 'O':
#                        false_neg += 1
#                elif p != y_:
#                    false_pos += 1
#        
#        prec = true_pos / (true_pos + false_pos) if true_pos + false_pos != 0 else 0
#        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg != 0 else 0
#        f1 = 2 * prec * recall / (prec + recall) if prec + recall != 0 else 0
#        return f1, prec, recall

    def generate_submission(self, test_preds, test_x, id_to_class, filename):
        test_seq_len = self.sess.run(self.seq_len, feed_dict={self.inputs: test_x})
        with open('./results/' + filename, 'w') as f:
            idx = 0
            for i in range(len(test_preds)):
                for j in range(test_seq_len[i]):
                    f.write('{}\n'.format(id_to_class[test_preds[i, j]]))
                    idx += 1
                f.write('\n')
        

if __name__ == '__main__':
    print('Start loading.')
    loader = Loader()
    train_x, train_pos, train_y = loader.load_data('train')
    dev_x, dev_pos, dev_y = loader.load_data('dev')
    test_x, test_pos, _ = loader.load_data('test')
    id_to_word = loader.id_to_word
    id_to_class = loader.id_to_class
    id_to_pos = loader.id_to_pos
    max_len = loader.max_len

    orth_loader = OrthLoader()
    train_orth = orth_loader.load_data('train')
    dev_orth = orth_loader.load_data('dev')
    test_orth = orth_loader.load_data('test')
    id_to_orth = orth_loader.id_to_word
    print('Done loading.')

    bilstm = BiLSTM(vocab_size=len(id_to_word), 
                    pos_vocab_size=len(id_to_pos),
                    orth_vocab_size=len(id_to_orth),
                    num_classes=len(id_to_class), 
                    sentence_len=max_len, 
                    embed_size=EMBED_SIZE, 
                    pos_embed_size=POS_EMBED_SIZE,
                    orth_embed_size=ORTH_EMBED_SIZE,
                    cell_size=CELL_SIZE,
                    num_layers=NUM_LAYERS)
    bilstm.train(train_x, train_pos, train_orth, train_y, id_to_class, 
                 num_epoch=NUM_EPOCH, batch_size=BATCH_SIZE)
    dev_f1, dev_prec, dev_rec = \
        bilstm.calculate_f1(dev_x, dev_pos, dev_orth, dev_y, id_to_class)

    print('[Dev]: F1 = {:.4f}, Prec = {:.4f}, Recall = {:.4f}'
          .format(dev_f1, dev_prec, dev_rec))
    logging.debug('[Dev]: F1 = {:.4f}, Prec = {:.4f}, Recall = {:.4f}'
                  .format(dev_f1, dev_prec, dev_rec))

    dev_preds = bilstm.predict(dev_x, dev_pos, dev_orth)
    bilstm.generate_submission(dev_preds, dev_x, id_to_class, 
                               filename=(FILENAME + '.dev'))

#    test_preds = bilstm.predict(test_x, test_pos, test_orth)
#    bilstm.generate_submission(test_preds, test_x, id_to_class, 
#                               filename=(FILENAME + '.test'))


