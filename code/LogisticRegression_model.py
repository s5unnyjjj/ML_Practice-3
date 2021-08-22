
import numpy as np


class l2_logistic():
    def __init__(self, num_iters=1000, alpha=0.01, lamb=0.1):
        self.num_iters = num_iters
        self.alpha = alpha
        self.lamb = lamb

    def sigmoid(self, x):
        sig_x = np.zeros(x.shape)
        sig_x = 1.0 / (1.0 + (np.exp(-x)))

        return sig_x

    def _cost(self, X, y, theta0, theta1, lamb):
        n = len(y)
        h = self.sigmoid(X.dot(theta1) + theta0)
        cal = y * np.log(h) + (1 - y) * np.log(1 - h)
        penalty = lamb * np.sum(np.power(theta1, 2))
        J = (-1 / n) * np.sum(cal) + penalty
        return J

    def _update(self, X, y, theta0, theta1, num_iters, alpha, lamb):
        n = len(y)
        J_all = np.zeros((num_iters, 1))

        for i in range(num_iters):
            h = self.sigmoid(X.dot(theta1) + theta0)

            theta0 = theta0 - alpha * (2 / n) * np.sum(h - y)
            theta1 = theta1 - alpha * (2 / n) * np.dot(X.T, (h - y)) - alpha * 2 * lamb * theta1

            J_all[i] = self._cost(X, y, theta0, theta1, lamb)

        return theta0, theta1, J_all

    def fit(self, X, y):
        self.theta1 = np.zeros((X.shape[1]))
        self.theta0 = 0
        self.J_all = np.zeros((self.num_iters, 1))

        self.theta0, self.theta1, self.J_all\
            = self._update(X, y, self.theta0, self.theta1, self.num_iters, self.alpha,
                                                            self.lamb)

    def predict(self, X):
        n = X.shape[0]
        probs = np.zeros(n)
        preds = np.zeros(n)
        pred_theta = X.dot(self.theta1) + self.theta0
        probs = self.sigmoid(pred_theta)
        idx = 0
        for val in probs:
            if val >= 0.5:
                preds[idx] = 1
            else:
                preds[idx] = 0
            idx += 1
        return probs, preds
