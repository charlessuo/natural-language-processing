#!/usr/bin/env python3
import numpy as np
import csv
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
import sys
import re

csv.field_size_limit(sys.maxsize)
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                      r'|^In article|^Quoted from|^\||^>)')

W2V_PATH = 'path/to/file'


class Loader:
    def __init__(self, dataset):
        if dataset == 'newsgroups':
            self.data_path = '../data/newsgroups/'
        else:
            raise ValueError('Not a valid dataset.')
        self.vocab_x = None
        self.vocab_y = None
        self.class_dict = None    

    def bow(self, ngram_range=(1, 1), dim_used=None):
        '''
        Bag-of-word feature for newsgroups classification.
        Args:
            ngram_range: tuple, the range for min to max grams used.
            dim_used: int, the number of maximum features for constructing bow model
        Returns:
            X_train, X_dev, X_test: ndarray of one-hot encoding with shape (#samples, len of vocabulary)
            Y_train, Y_dev: labels in vector form with class ids (not one-hot), shape (#samples,)
        Store:
            vocab_x: dict, mapping from words to indices
        '''
        texts_train, Y_train = self._load_data('train')
        texts_dev, Y_dev = self._load_data('dev')
        texts_test, _ = self._load_data('test')
        with open('stopwords.txt') as f:
            stopwords = [word.strip() for word in f.readlines()]
        vectorizer = CountVectorizer(max_features=dim_used, 
                                     ngram_range=ngram_range, 
#                                     stop_words=stopwords,
#                                     tokenizer=self._stem_tokenize, 
                                     binary=True)
        X_train = vectorizer.fit_transform(texts_train).toarray()
        X_dev = vectorizer.transform(texts_dev).toarray()
        X_test = vectorizer.transform(texts_test).toarray()
        self.vocab_x = vectorizer.vocabulary_
        return X_train, Y_train, X_dev, Y_dev, X_test

    def tfidf(self, ngram_range=(1, 1), dim_used=None):
        '''
        Bag-of-word feature for newsgroups classification.
        Args:
            ngram_range: tuple, the range for min to max grams used.
            dim_used: int, the number of maximum features for constructing bow model
        Returns:
            X_train, X_dev, X_test: ndarray of one-hot encoding with shape (#samples, len of vocabulary)
            Y_train, Y_dev: labels in vector form with class ids (not one-hot), shape (#samples,)
        Store:
            vocab_x: dict, mapping from words to indices
        '''
        texts_train, Y_train = self._load_data('train')
        texts_dev, Y_dev = self._load_data('dev')
        texts_test, _ = self._load_data('test')
        with open('stopwords.txt') as f:
            stopwords = [word.strip() for word in f.readlines()]
        vectorizer = TfidfVectorizer(max_features=dim_used, 
#                                     tokenizer=self._stem_tokenize,
#                                     stop_words=stopwords,
                                     ngram_range=ngram_range)
        X_train = vectorizer.fit_transform(texts_train).toarray()
        X_dev = vectorizer.transform(texts_dev).toarray()
        X_test = vectorizer.transform(texts_test).toarray()
        self.vocab_x = vectorizer.vocabulary_
        return X_train, Y_train, X_dev, Y_dev, X_test

    def average_w2v(self):
        model = gensim.models.Word2Vec.load_word2vec_format(W2V_PATH, binary=True)
        print('Done loading word2vec model.')

        texts_train, Y_train = self._load_data('train')
        texts_dev, Y_dev = self._load_data('dev')
        texts_test, _ = self._load_data('test')
        with open('stopwords.txt') as f:
            stopwords = [word.strip() for word in f.readlines()]

        X_train = np.zeros((texts_train.shape[0], 300))
        X_dev = np.zeros((texts_dev.shape[0], 300))
        X_test = np.zeros((texts_test.shape[0], 300))
        datasets = [X_train, X_dev, X_test]

        for i, texts in enumerate([texts_train, texts_dev, texts_test]):
            for j, text in enumerate(texts):
                words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
                num_words = 0
                for word in words:
                    if word in stopwords:
                        continue
                    if word in model:
                        datasets[i][j] += model[word]
                        num_words += 1
                datasets[i][j] /= num_words
        
        return X_train, Y_train, X_dev, Y_dev, X_test

    def bow_(self, dim_used=5000):
        '''
        Bag-of-word feature for newsgroups classification.
        Args:
            None
        Returns:
            X_train, X_dev, X_test: ndarray of one-hot encoding with shape (#samples, len of vocabulary)
            Y_train, Y_dev: labels in vector form with class ids (not one-hot), shape (#samples,)
        Store:
            vocab_x: dict, mapping from words to indices
        '''
        texts_train, Y_train = self._load_data('train')
        texts_dev, Y_dev = self._load_data('dev')
        texts_test, _ = self._load_data('test')
        vocab_x = {}
        datasets = []

        # build vocab_x        
        for texts in [texts_train, texts_dev, texts_test]:
            X_tmp = []
            idx = 0
            with open('stopwords.txt') as f:
                stopwords = set([w.strip() for w in f.readlines()])
            for text in texts:
                words = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize(text) if w.isalpha()]
                X_tmp.append([])
                for word in words:
                    if word in stopwords:
                        continue
                    if self.vocab_x is None and word not in vocab_x:
                        vocab_x[word] = idx
                        idx += 1
                    X_tmp[-1].append(word)
            self.vocab_x = vocab_x
            datasets.append(X_tmp)

        # generate data
        X_train = np.zeros((texts_train.shape[0], len(vocab_x)))
        X_dev = np.zeros((texts_dev.shape[0], len(vocab_x)))
        X_test = np.zeros((texts_test.shape[0], len(vocab_x)))

        for X_tmp, X in zip(datasets, [X_train, X_dev, X_test]):
            for i in range(len(X_tmp)):
                for word in X_tmp[i]:
                    if word in vocab_x:
                        X[i][vocab_x[word]] = 1 # presence/absence
        return X_train[:, :dim_used], Y_train, X_dev[:, :dim_used], Y_dev, X_test[:, :dim_used]
    
    def _load_data(self, mode):
        '''
        Load data.
        Args:
            mode: str, 'train' or 'dev' or 'test'
        Returns:
            texts: ndarray of strings with shape (#samples,)
            Y: labels in vector form with class ids (not one-hot), shape (#samples,)
        Store:
            vocab_y: dict, mapping from classes to indices
            class_dict: dict, mapping from indices to classes
        '''
        data_path = self.data_path
        texts = [] 
        with open(data_path + mode + '/' + mode + '_data.csv') as f_data:
            next(f_data)
            reader = csv.reader(f_data)
            for text in reader:
                text = text[1]
                #text = self._strip_header(text)
                #text = self._strip_quote(text)
                #text = self._strip_footer(text)
                texts.append(text)
        texts = np.array(texts)
        
        if mode == 'test':
            Y = None
        elif mode == 'train':
            labels = []
            with open(data_path + mode + '/' + mode + '_labels.csv') as f_label:
                next(f_label)
                reader = csv.reader(f_label)
                for line in reader:
                    labels.append(line[1])
            labels = np.array(labels)
            vocab_y = {k: v for k, v in zip(list(set(labels)), range(len(labels)))}
            class_dict = {k: v for k, v in zip(range(len(labels)), list(set(labels)))}
            self.vocab_y = vocab_y
            self.class_dict = class_dict
        
            Y = np.zeros_like(labels)
            for i in range(len(labels)):
                Y[i] = vocab_y[labels[i]]
            Y = Y.astype(int)
        elif mode == 'dev':
            if not self.vocab_y:
                raise ValueError('Training data not loaded yet (no vocab available).')
            vocab_y = self.vocab_y
            labels = []
            with open(data_path + mode + '/' + mode + '_labels.csv') as f_label:
                next(f_label)
                reader = csv.reader(f_label)
                for line in reader:
                    labels.append(line[1])
            labels = np.array(labels)
            Y = np.zeros_like(labels)
            for i in range(len(labels)):
                Y[i] = vocab_y[labels[i]]
            Y = Y.astype(int) 
        else:
            raise ValueError('Not a valid mode.')
        return texts, Y

    def _stem_tokenize(self, text):
        tokens = [stemmer.stem(item) for item in word_tokenize(text) if item.isalpha()]
        return tokens

    def _strip_header(self, text):
        _before, _blankline, after = text.partition('\n\n')
        return after

    def _strip_quote(self, text):
        '''
        Given text in "news" format, strip lines beginning with the quote
        characters > or |, plus lines that often introduce a quoted section
        (for example, because they contain the string 'writes:'.)
        '''
        good_lines = [line for line in text.split('\n') if not QUOTE_RE.search(line)]
        return '\n'.join(good_lines)

    def _strip_footer(self, text):
        '''
        Given text in "news" format, attempt to remove a signature block.
        As a rough heuristic, we assume that signatures are set apart by either
        a blank line or a line made of hyphens, and that it is the last such line
        in the file (disregarding blank lines at the end).
        '''
        lines = text.strip().split('\n')
        for line_num in reversed(range(len(lines))):
            line = lines[line_num]
            if line.strip().strip('-') == '':
                break 
        if line_num > 0:
            return '\n'.join(lines[:line_num])
        else:
            return text


if __name__ == '__main__':

    loader = Loader('newsgroups')
    X_train, Y_train, X_dev, Y_dev, X_test = loader.bow(ngram_range=(1, 1), dim_used=5000)

    print('X_train:', X_train.shape)
    print('Y_train', Y_train.shape)
    print('X_dev:', X_dev.shape)
    print('Y_dev:', Y_dev.shape)
    print('X_test:', X_test.shape)
    print('length of vocab_x:', len(loader.vocab_x))
#    print('vocab_x:', loader.vocab_x)
    print('vocab_y:', loader.vocab_y)
    print('vocab_y:', loader.class_dict)

