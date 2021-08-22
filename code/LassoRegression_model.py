
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


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


def crossValidation_Lasso(lambdas, num_fold, X_train, y_train, X_test, y_test):
    y_train = y_train.values
    MSE_set = []
    best_MSE = 0.
    best_lambda = 0.
    test_MSE = 0.
    lasso = None


    for lamb in lambdas:
        MSE_fold = 0.

        lasso = Lasso(lamb, max_iter=10000)

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

            lasso.fit(trainX_set, trainY_set)
            pred = lasso.predict(valX_set)

            MSE_fold += mean_squared_error(valY_set, pred)
        MSE_value = MSE_fold / num_fold

        MSE_set.append(MSE_value)
        best_MSE = min(MSE_set)

        if MSE_value == best_MSE:
            best_lambda = lamb

    lasso.set_params(alpha=best_lambda)
    lasso.fit(X_train, y_train)

    test_pred = lasso.predict(X_test)
    test_MSE = mean_squared_error(y_test, test_pred)

    return MSE_set, best_MSE, best_lambda, test_MSE, lasso

