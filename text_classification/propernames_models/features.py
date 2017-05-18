#!/usr/bin/env python3
import numpy as np
import csv
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys

csv.field_size_limit(sys.maxsize)
lemmatizer = WordNetLemmatizer()


class Loader:
    def __init__(self, dataset):
        if dataset == 'propernames':
            self.data_path = '../data/propernames/'
        else:
            raise ValueError('Not a valid dataset.')
        self.vocab_x = None
        self.vocab_y = None
        self.class_dict = None

    def char_ngram(self, ngram_range=(2, 2), dim_used=None):
        texts_train, Y_train = self._load_data('train')
        texts_dev, Y_dev = self._load_data('dev')
        texts_test, _ = self._load_data('test')
        vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range, max_features=dim_used)
        X_train = vectorizer.fit_transform(texts_train).toarray()
        X_dev = vectorizer.transform(texts_dev).toarray()
        X_test = vectorizer.transform(texts_test).toarray()
        self.vocab_x = vectorizer.vocabulary_
        return X_train, Y_train, X_dev, Y_dev, X_test
    
    def char_ngram_(self, n=2):
        '''
        Character n-grams for propernames classification.
        Args:
            n: int, n-gram
        Return:
            X_train, X_dev, X_test: ndarray of one-hot encoding with shape (#samples, len of vocabulary)
            Y_train, Y_dev: labels in vector form with class ids (not one-hot), shape (#samples,)
        Store:
            vocab_x: dict, mapping from grams to indices
        '''
        texts_train, Y_train = self._load_data('train')
        texts_dev, Y_dev = self._load_data('dev')
        texts_test, _ = self._load_data('test')
        
        # build vocab_x
        vocab_x = {}
        datasets = []
        for texts in [texts_train, texts_dev, texts_test]:
            X_tmp = []
            idx = 0
            for text in texts:
                grams = list(ngrams([char.lower() for char in text], n))
                X_tmp.append(grams)
                if self.vocab_x is not None:
                    continue
                for gram in grams:
                    if gram not in vocab_x:
                        vocab_x[gram] = idx
                        idx += 1
            self.vocab_x = vocab_x
            datasets.append(X_tmp)
        
        # generate data
        X_train = np.zeros((texts_train.shape[0], len(vocab_x)))
        X_dev = np.zeros((texts_dev.shape[0], len(vocab_x)))
        X_test = np.zeros((texts_test.shape[0], len(vocab_x)))
        for X_tmp, X in zip(datasets, [X_train, X_dev, X_test]):
            for i in range(len(X_tmp)):
                for gram in X_tmp[i]:
                    if gram in vocab_x:
                        X[i][vocab_x[gram]] += 1 # count
        return X_train, Y_train, X_dev, Y_dev, X_test 
    
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
            for line in reader:
                texts.append(line[1])
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


if __name__ == '__main__':
    loader = Loader('propernames')
    X_train, Y_train, X_dev, Y_dev, X_test = loader.char_ngram(ngram_range=(1, 5), dim_used=5000)

    print('X_train:', X_train.shape)
    print('Y_train', Y_train.shape)
    print('X_dev:', X_dev.shape)
    print('Y_dev:', Y_dev.shape)
    print('X_test:', X_test.shape)
    print('length of vocab_x:', len(loader.vocab_x))
#    print('vocab_x:', loader.vocab_x)
    print('vocab_y:', loader.vocab_y)
    print('vocab_y:', loader.class_dict)

