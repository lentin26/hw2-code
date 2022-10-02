from numpy.random import choice
from numpy import matmul
from numpy import zeros
from numpy import ones
from numpy import exp


class LogisticRegression:
    """
    Logistic Regression classifier
    """
    def __init__(self, X, y) -> None:
        # data
        self.X = X
        # target
        self.y = y
        # data dimensions
        self.n = X.shape[0]
        self.d = X.shape[1]
        # probabilities
        self.p = zeros(len(y))
        self.p[y == 0] = (y == 0).sum()/len(y)
        self.p[y == 1] = (y == 1).sum()/len(y)
        # weights
        self.w = ones(self.d)

    def update_weights(self, i):
        """
        Update weight for a point x
        """
        # variance
        R = self.p[i] * (1 - self.p[i])
        # effective target value
        z = (self.X[i, :] * self.w).sum() - (1/R) * (self.p[i] - self.y[i])

        self.w = self.X[i, :] * z/(self.X[i, :] * self.X[i, :]).sum()

    def fit(self, max_iter):
        """
        fit model to data using stochastic gradient descent
        """
        # start training
        for _ in range(max_iter):
            # randomly sample point
            i = choice(self.n)
            # update weights
            self.update_weights(i)

    def predict_proba(self, X):
        """
        following fit return predictions probabilities given by sigmoid function
        """
        return 1/(1 - exp(matmul(X, self.w)))

    def predict_class(self, X):
        """
        Assign class to instances
        :return:
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)


if __name__ == '__main__':
    import numpy as np

    # set seed
    np.random.seed(143)

    # multivariate normal data in 2d
    size = 1000
    data1 = np.random.multivariate_normal([1, 1], np.identity(2), size=size)
    data2 = np.random.multivariate_normal([-1, -1], np.identity(2), size=size)

    # prep data
    X = np.concatenate([data1, data2], axis=0)
    y = np.array([0]*size + [1]*size)

    lr = LogisticRegression(X=X, y=y)
    max_iter=100
    lr.fit(max_iter=max_iter)

    y_predict = lr.predict_proba(X)
    y_predict
