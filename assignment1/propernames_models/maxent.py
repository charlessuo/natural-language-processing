#!/usr/bin/env python3
import scipy.optimize as opt
from features import Loader


class MaxEnt:
    def __init__(self):
        self.W = None
        self.N = None
        self.block_size = None
        self.num_classes = None

    def train(self, X, Y, lmda=1):
        self.N, self.block_size = X.shape
        self.num_classes = max(Y) + 1
        self.W = np.zeros(self.block_size * self.num_classes)
        
        # start optimization
        self.W, loss, info = opt.fmin_l_bfgs_b(
            self._log_likelihood, 
            x0=self.W,
            args=(X, Y, lmda), 
            fprime=self._log_likelihood_grad, 
            approx_grad=False)
        print('Loss:', loss)
        print('Optimization Info:', info)

    def _log_likelihood(self, para, *args):
        X, Y, lmda = args
        W = para
        W_ = W.reshape(self.num_classes, self.block_size)
#        L = np.sum(W_[Y] * X) - np.sum(np.log(np.sum(np.exp(np.dot(X, W_.T)), axis=1))) #- 0.5 * lmda * np.sum(W_ ** 2)
        probs = self._softmax(np.dot(X, W_.T))
        L = np.sum(np.log([probs[i][Y[i]] for i in range(self.N)])) - 0.5 * lmda * np.sum(W_ ** 2)
        return -L

    def _log_likelihood_grad(self, para, *args):
        X, Y, lmda = args
        W = para
        W_ = W.reshape(self.num_classes, self.block_size)
        dL = np.zeros_like(W_)
        probs = self._softmax(np.dot(X, W_.T))
        
        for i in range(self.num_classes):
            dL[i] += np.sum(X[Y == i], axis=0) - np.sum(X * probs[:, i].reshape(-1, 1), axis=0)
        dL -= lmda * W_ # regularization
        return -dL.flatten()

    def _softmax(self, M):
        '''
        Calculate softmax.
        Args:
            M: Each row contains an inner products vector for all classes (#smaple, num_classes)
               The softmax will be applied on the rows.
        Returns:
            M: The probability score matrix (along rows)
        '''
        if M.ndim > 1:
            M -= np.max(M, axis=1).reshape(M.shape[0], 1)
            M = np.exp(M) / np.sum(np.exp(M), axis=1).reshape(M.shape[0], 1)
        else:
            M -= np.max(M)
            M = np.exp(M) / np.sum(np.exp(M))
        return M

    def predict(self, X_test):
        W_ = self.W.reshape(self.num_classes, self.block_size)
        Y_ = np.argmax(np.dot(W_, X_test.T), axis=0)
        return Y_

    def score(self, X_dev, Y_dev):
        Y_ = self.predict(X_dev)
        acc = np.sum(Y_ == Y_dev) / len(Y_dev)
        return acc


def generate_submission(Y_pred, class_dict, filename='submission'):
    with open('./results/' + filename + '.csv', 'w') as f:
        f.write('id,type\n')
        for i in range(len(Y_pred)):
            f.write('{},{}\n'.format(str(i), class_dict[Y_pred[i]]))


if __name__ == '__main__':
    loader = Loader('propernames')
    X_train, Y_train, X_dev, Y_dev, X_test = loader.char_ngram(ngram_range=(2, 3), dim_used=3000)
    print('Done loading data.')

    maxent = MaxEnt()
    maxent.train(X_train, Y_train)
    print('Done training.')

    acc = maxent.score(X_dev, Y_dev)
    print('Dev set accuracy:', acc)

#    Y_pred = maxent.predict(X_test)
#    generate_submission(Y_pred, loader.class_dict, filename='maxent_n3')

