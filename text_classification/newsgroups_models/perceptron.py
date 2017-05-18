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


def generate_submission(Y_pred, class_dict, filename='submission'):
    with open('./results/' + filename + '.csv', 'w') as f:
        f.write('id,newsgroup\n')
        for i in range(len(Y_pred)):
            f.write('{},{}\n'.format(str(i), class_dict[Y_pred[i]]))


if __name__ == '__main__':
    loader = Loader('newsgroups')
    X_train, Y_train, X_dev, Y_dev, Y_test = loader.bow()
    print('Done loading data.')

    perceptron = Perceptron()
    perceptron.train(X_train[:, :10000], Y_train, MAXITER=20000)

    acc = perceptron.score(X_dev[:, :10000], Y_dev)
    print('Dev set accuracy:', acc)

    Y_pred = perceptron.predict(X_test)
    generate_submission(Y_pred, loader.class_dict, 'test')

