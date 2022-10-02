from numpy import argmax
from numpy.linalg import norm


class GaussainKernel():
    """
    Gaussian kernel density estimator.
    """
    def __init__(self, h):
        # gaussian variance
        self.h = h

    def fit(self, X_train, y_train):
        """
        Fit model to training set
        :param X_train: training data
        :param y_train: training labels
        :return: None
        """
        self.X_train = X_train
        self.y_train = y_train

    def get_likelihood(self, X_test, c):
        """
        likelihood of assigning points in X_test to c given X_train
        :param c: training set class
        :param X_test: test data
        :return: likelihood
        """
        # get training data corresponding to class c
        X_train = self.X_train[self.y_train == c]
        # shape parameters
        N = X_train.shape[0]
        D = X_train.shape[1]
        # compute likelihood
        z = 0
        for i in range(N):
            z += (1/(2*np.pi*self.h**2))**(D/2)*np.exp(-1*norm(X_test - X_train[i, :], axis=1)/(2*self.h**2))

        # normalize and return likelihood
        return z/N

    def predict(self, X_test):
        """
        Train a parzen model
        :param X_test: test data
        :return: classification of points in X_test
        """
        post_proba = []
        for c in np.unique(self.y_train):
            # compute prior
            prior = (y == c).sum()/len(y)
            # compute likelihood
            likelihood = self.get_likelihood(X_test=X_test, c=c)
            # compute posterior
            post_proba.append(prior*likelihood)

        return argmax(np.array(post_proba).T, axis=1)


if __name__ == '__main__':
    import numpy as np

    # generate checker board data
    def gen_cb(N, a, alpha):
        """
        N: number of points on the checkerboard
        a: width of the checkerboard (0<a<1)
        alpha: rotation of the checkerboard in radians
        """
        d = np.random.rand(N, 2).T
        d_transformed = np.array([d[0] * np.cos(alpha) - d[1] * np.sin(alpha),
                                  d[0] * np.sin(alpha) + d[1] * np.cos(alpha)]).T
        s = np.ceil(d_transformed[:, 0] / a) + np.floor(d_transformed[:, 1] / a)
        lab = 2 - (s % 2)
        data = d.T
        return data, lab


    N = 1000
    X, y = gen_cb(N=N, a=0.5, alpha=0)

    # split data into 20% test and 80% train
    k = int(0.80 * N)
    X_train = X[:k, :]
    y_train = y[:k]
    X_test = X[k:, :]
    y_test = y[k:]

    # instantiate classifier
    clf = GaussainKernel(h=1)

    # fit model
    clf.fit(X_train, y_train)

    # predict labels on test dataset
    clf.predict(X_test)
