#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import random
import time
from nltk.stem.porter import PorterStemmer
from scipy import stats
from loader import Loader

stemmer = PorterStemmer()


class Word2Vec:
    def __init__(self, vocab, rev_vocab, batch_size=128, embed_size=100, 
                 num_sampled=64, full_window_size=3, num_steps=None):
        self.vocab = vocab
        self.rev_vocab = rev_vocab
        self.batch_size = batch_size
        self.vocab_size = len(vocab)
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.full_window_size = full_window_size
        self.num_steps = num_steps
        if num_steps is None:
            raise ValueError('Number of steps should be specified.')
                
        self.train_x = tf.placeholder(tf.int32, shape=[self.batch_size], name='train_x')
        self.train_y = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='train_y')
        self.sess = tf.Session()

        with tf.name_scope('embedding'), tf.device('/cpu:0'):
            self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0), name='embeddings')
            self.embedded_x = tf.nn.embedding_lookup(self.embeddings, self.train_x, name='embedded_x')

        with tf.name_scope('output'):
            self.W = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size], 
                                                     stddev=1.0 / np.sqrt(self.embed_size)), 
                                                     name='W')
            self.b = tf.Variable(tf.zeros([self.vocab_size]), 'b')

            # different losses: sampled_softmax_loss, nce_loss
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=self.W, 
                                                                  biases=self.b, 
                                                                  inputs=self.embedded_x,
                                                                  labels=self.train_y,
                                                                  num_sampled=self.num_sampled, 
                                                                  num_classes=self.vocab_size,
                                                                  name='loss'))

    def train(self, data):
        '''
        Set up optimizer and start training.
        '''
        # set optimizer
        optimizer = tf.train.AdagradOptimizer(1.0)
        train_op = optimizer.minimize(self.loss)

        # normalize embeddings
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.norm_embeddings = self.embeddings / norm
        
        # determine maximum step 
        self.idx_hop = int(self.batch_size / (self.full_window_size - 1))
        if self.num_steps * self.idx_hop < len(data):
            termination = self.num_steps * self.idx_hop
            print('Run {} steps.'.format(int(self.num_steps)))
        else:
            termination = len(data)
            print('Assigned number of steps exceeded data length. ' 
                  'Use data length instead. Run ~{} steps.'.format(int(len(data) / self.idx_hop)))
        batch_pairs = self._extract_batch(data, termination)

        self.sess.run(tf.global_variables_initializer())
        step = 0
        avg_loss = 0
        for batch_x, batch_y in batch_pairs:
            _, loss_ = self.sess.run([train_op, self.loss], 
                                     feed_dict={self.train_x: batch_x, self.train_y: batch_y})
            step += 1
            avg_loss += loss_
            if step % 2000 == 0:
                print('Step {}, Average Loss: {:.6f}'.format(step, avg_loss / 1000))
                avg_loss = 0

    def _extract_batch(self, data, termination):
        '''
        Extract batch for each step with assigned batch size.
        Args:
            data: list, the whole text data.
            termination: int, stopping criterion
        Yields:
            centers: ndarray with shape (batch size,), center word ids
            contexts: ndarray with shape (batch size, 1), corresponding context word ids
        '''
        half_window = int(self.full_window_size / 2)
        context_len = 2 * half_window

        idx = half_window
        while idx + half_window + self.idx_hop < termination:
            centers = np.zeros((self.batch_size,), dtype=np.int32)
            contexts = np.zeros((self.batch_size, 1), dtype=np.int32)
            for window_idx in range(self.idx_hop):
                for i in range(self.full_window_size):
                    if i == half_window:
                        continue
                    elif i < half_window:
                        centers[i + window_idx * context_len] = data[window_idx + idx]
                        contexts[i + window_idx * context_len, 0] = data[i - half_window + window_idx + idx]
                    elif i > half_window:
                        centers[i - 1 + window_idx * context_len] = data[window_idx + idx]
                        contexts[i - 1 + window_idx * context_len, 0] = data[i - half_window + window_idx + idx]
                    else:
                        raise ValueError('Should not happen.')
            idx += self.idx_hop
            yield centers, contexts

    def evaluate(self, counts, stems, word_pairs, simu_labels=None, filename=None):
        '''
        Evaluate on dev set or generate correlations on test set for submission.
        Args:
            counts: dict, word to count mapping
            stems: dict, stemmed word to original word mapping
            word_pairs: list of list, word pairs to evaluate on
            simu_labels: list, dev set correlation labels
            filename: str, file name for submission
        Returns:
            None
        '''
        word_id_pairs = []
        uncommon_words = [word for word in counts.keys() if counts[word] == 1] # sample uncommon words
        for i, word_pair in enumerate(word_pairs):
            id_pair = []
            for w, uncommon_word in zip(word_pair, np.random.choice(uncommon_words, 2, replace=False)):
                if w in self.vocab:
                    # if the word is found, use it to lookup
                    id_ = self.vocab[w]
                else:
                    stemmed_word = stemmer.stem(w)
                    if stemmed_word in stems:
                        # if the word is not found, find its stemmed form
                        id_ = self.vocab[stems[stemmed_word]]
                        print('{}: Word "{}" found as "{}"'.format(i, w, stemmed_word))
                    else:
                        # if not found in vocab and stem dict, treat it as a uncommom word
                        id_ = self.vocab[uncommon_word]
                        print('{}: Word "{}" is NOT found'.format(i, w))
                id_pair.append(id_)
            word_id_pairs.append(id_pair)

        # look up for words
        word_id_flat = np.array(word_id_pairs).flatten()
        words = tf.placeholder(tf.int32, shape=[len(word_id_flat)], name='words')
        embedded_words = tf.nn.embedding_lookup(self.norm_embeddings, words, name='embedded_words')
        wordvecs = self.sess.run(embedded_words, feed_dict={words: word_id_flat})
        simu_predicts = np.sum(wordvecs[::2] * wordvecs[1::2], axis=1)

        if simu_labels:
            # evaluate on dev set
            corr = stats.spearmanr(simu_predicts, simu_labels).correlation
            print('Correlation:', corr)
        else:
            # generate submission file
            with open('./submissions/' + filename + '.csv', 'w') as f:
                f.write('id,similarity\n')
                for i, similarity in enumerate(simu_predicts):
                    f.write('{},{}\n'.format(i, similarity))

    def export_embeddings(self):
        '''
        Export the whole embedding table to a file.
        '''
        ids = list(range(self.vocab_size))
        all_ids = tf.placeholder(tf.int32, shape=[self.vocab_size], name='all_ids')
        embedded_words = tf.nn.embedding_lookup(self.norm_embeddings, all_ids, name='embedded_words')
        id_to_embedding = self.sess.run(embedded_words, feed_dict={all_ids: ids}) # shape=(vocab_size, embed_size)
        print('Final embedding shape: ', id_to_embedding.shape)

        with open('embeddings.txt', 'w') as f:
            for i, embed_vec in enumerate(id_to_embedding):
                vec_str = ''
                for num in embed_vec:
                    vec_str += str(num) + ' '
                f.write('{} {}\n'.format(self.rev_vocab[ids[i]], vec_str)) 


if __name__ == '__main__':
    start = time.time()
    loader = Loader()
    data, vocab, rev_vocab, counts, stems = loader.load_data('data3m')
    dev_word_pairs, simu_labels = loader.load_eval()
    test_word_pairs = loader.load_test()
    print('Done loading data.')
    print('Data size:', len(data))
    print('Vocabulary size:', len(vocab))

    w2v = Word2Vec(vocab, rev_vocab, embed_size=50, full_window_size=3, num_steps=5e+5)
    w2v.train(data)
    w2v.evaluate(counts, stems, dev_word_pairs, simu_labels=simu_labels)
    w2v.evaluate(counts, stems, test_word_pairs, filename='submission.csv')

    print('Time used: {} min'.format(int((time.time() - start) / 60)))

