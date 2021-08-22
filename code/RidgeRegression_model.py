
import numpy as np
from sklearn.metrics import mean_squared_error


class ridgeRegression():
    def __init__(self, num_iters=1000, alpha=0.003, lamb=0.1):
        self.num_iters = num_iters
        self.alpha = alpha
        self.lamb = lamb

    def set_params(self, lamb):
        self.lamb = lamb

    def _cost(self, X, y, theta0, theta1, lamb):
        n = X.shape[0]
        J = np.inf
        pred_y = X.dot(theta1) + theta0

        J1 = np.sum(np.power((pred_y-y), 2)) / n
        J2 = lamb * np.sum(np.power(theta1, 2))
        J = J1 + J2
        return J

    def _update(self, X, y, theta0, theta1, num_iters, alpha, lamb):
        n = X.shape[0]
        J_all = np.zeros((num_iters, 1))
        for i in range(num_iters):
            pred_y = X.dot(theta1) + theta0

            theta0 = theta0 - alpha * (2/n) * np.sum(pred_y-y)
            theta1 = theta1 - alpha * (2/n) * np.dot(X.T, (pred_y-y)) - alpha*2*lamb*theta1

            J_all[i] = self._cost(X, y, theta0, theta1, lamb)
        return theta0, theta1, J_all

    def fit(self, X, y):
        self.theta1 = np.zeros(X.shape[1])
        self.theta0 = 0
        self.J_all = np.zeros((self.num_iters, 1))

        self.theta0, self.theta1, self.J_all \
            = self._update(X, y, self.theta0, self.theta1, self.num_iters, self.alpha, self.lamb)

    def predict(self, X):
        pred = X.dot(self.theta1) + self.theta0
        return pred


def ridgeClosed_get_theta(X, y, lamb):
    n = len(y)
    theta_0 = 0
    theta_1 = np.zeros(X.shape[1])
    I = np.identity(X.shape[1])
    theta_0 = (1 / n) * np.sum(y)

    theta_1 = np.dot(np.linalg.inv((np.dot(X.T, X) / n) + lamb * I), (np.dot(X.T, y) / n))
    return theta_0, theta_1


def ridgeClosed_predict(X, theta_0, theta_1):
    pred = X.dot(theta_1) + theta_0
    return pred


def kFold(X, n_splits, random_state=9):
    train_index = []
    val_index = []

    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.RandomState(random_state).shuffle(indices)
    fold_size = n_samples // n_splits

    for k in range(n_splits):
        val_start = k * fold_size
        val_end = (k+1) * fold_size
        train_start = k * (n_samples-fold_size) + val_end
        if k == 0:
            val_index.extend(indices[val_start:val_end])
            train_index.extend(indices[train_start:])
        elif k == (n_splits-1):
            val_index.extend(indices[val_start:val_end])
            train_index.extend(indices[:val_start])
        else:
            train_index.extend(indices[:val_start])
            val_index.extend(indices[val_start:val_end])
            train_index.extend(indices[val_end:])
    train_index = np.array(train_index)
    val_index = np.array(val_index)

    return train_index, val_index


def crossValidation_Ridge(lambdas, num_fold, X_train, y_train, X_test, y_test):
    y_train = y_train.values
    MSE_set = []
    best_MSE = 0.
    best_lambda = 0.
    test_MSE = 0.
    ridge = None

    for lamb in lambdas:
        MSE_fold = 0.

        ridge = ridgeRegression(lamb=lamb)

        total_train_idx, total_val_idx = kFold(X_train, num_fold)
        val_size = len(X_train) // num_fold
        train_size = len(X_train) - val_size

        for i in range(num_fold):
            val_start = i * val_size
            val_end = (i + 1) * val_size
            train_start = i * train_size
            train_end = (i + 1) * train_size

            train_idx = total_train_idx[train_start:train_end]
            val_idx = total_val_idx[val_start:val_end]

            trainX_set = np.zeros((len(train_idx), X_train.shape[1]))
            trainY_set = list()
            valX_set = np.zeros((len(val_idx), X_train.shape[1]))
            valY_set = list()

            for j in range(len(train_idx)):
                trainX_set[j] = X_train[train_idx[j]]
                trainY_set.append(y_train[train_idx[j]])
            for k in range(len(val_idx)):
                valX_set[k] = X_train[val_idx[k]]
                valY_set.append(y_train[val_idx[k]])

            ridge.fit(trainX_set, trainY_set)
            pred = ridge.predict(valX_set)

            MSE_fold += mean_squared_error(valY_set, pred)
        MSE_value = MSE_fold/num_fold

        MSE_set.append(MSE_value)
        best_MSE = min(MSE_set)

        if MSE_value == best_MSE:
            best_lambda = lamb

    ridge.set_params(best_lambda)
    ridge.fit(X_train, y_train)

    test_pred = ridge.predict(X_test)
    test_MSE = mean_squared_error(y_test, test_pred)

    return MSE_set, best_MSE, best_lambda, test_MSE, ridge