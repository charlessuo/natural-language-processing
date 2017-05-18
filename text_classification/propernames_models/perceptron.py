#!/usr/bin/env python3
import numpy as np
from features import Loader


class Perceptron:
    def __init__(self):
        self.W = None
    
    def train(self, X, Y, MAXITER=1000, lr=0.01):
        # set weights with padding
        self.W = np.zeros((X.shape[1] + 1, Y.shape[0]))

        # pad ones to inputs
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # run training
        for i in range(MAXITER):
            idx = i % X.shape[0]
            if idx % 500 == 0:
                print('Epoch: {}, Step: {}'.format(i // X.shape[0] + 1, idx))

            y_ = np.argmax(np.dot(X[idx], self.W))
            if y_ != Y[idx]:
                self.W[:, y_] -= lr * X[idx]
                self.W[:, Y[idx]] += lr * X[idx]

    def predict(self, X_test):
        X_test = np.hstack((np.ones((X_dev.shape[0], 1)), X_test))
        Y_ = np.argmax(np.dot(X_test, self.W), axis=1)
        return Y_

    def score(self, X_dev, Y_dev):
        Y_ = self.predict(X_dev)
        acc = np.sum(Y_dev == Y_) / Y_dev.shape[0]
        return acc


if __name__ == '__main__':
    loader = Loader('propernames')
    X_train, Y_train, X_dev, Y_dev, Y_test = loader.char_ngram(n=2)
    print('Done loading data.')

    perceptron = Perceptron()
    perceptron.train(X_train, Y_train, MAXITER=40000)
    acc = perceptron.score(X_dev, Y_dev)
    print('Dev set accuracy:', acc)


