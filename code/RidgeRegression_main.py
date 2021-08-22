

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from RidgeRegression_model import *
from utils import *


def dataProcessing():
    df = pd.read_csv('./data1.csv').dropna().drop('Unnamed: 0', axis=1)

    dummies = pd.get_dummies(df[['League', 'Division', 'NewLeague']])

    y = df.Salary
    X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
    X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=False)

    standardizer = StandardScaler()
    scaled_X_train = standardizer.fit_transform(X_train)
    scaled_X_test = standardizer.transform(X_test)

    return X_train, y_train, X_test, y_test, scaled_X_train, scaled_X_test, X.columns

if __name__ == "__main__":
    trainX, trainY, testX, testY, scaled_trainX, scaled_testX, colX = dataProcessing()

    lamb = 0
    linear_regression = ridgeRegression(lamb=lamb)
    linear_regression.fit(scaled_trainX, trainY)
    pred = linear_regression.predict(scaled_testX)
    print('lambda {:6.3f} - MSE : {:9.2f}'.format(lamb, mean_squared_error(testY, pred)))

    lambdas = 10 ** np.linspace(2, -2, 50)
    coefs = np.zeros((50, 19))

    for index, lamb in enumerate(lambdas):
        ridge = ridgeRegression(lamb=lamb)
        ridge.fit(scaled_trainX, trainY)
        coefs[index] = ridge.theta1
    vis_coef(lambdas, coefs, method='Ridge')

    lambs = [0.001, 0.01, 0.1, 1., 10.]
    for lamb in lambs:
        ridge2 = ridgeRegression(alpha=0.003, lamb=lamb)
        ridge2.fit(scaled_trainX, trainY)
        pred = ridge2.predict(scaled_testX)
        print('lambda {:6.3f} - MSE : {:9.2f}'.format(lamb, mean_squared_error(testY, pred)))

    lambdas = 10 ** np.linspace(1, -1.5, 20) * 0.4
    MSE_set, best_MSE, best_lambda, test_MSE, ridge = crossValidation_Ridge(lambdas, 5, scaled_trainX, trainY,
                                                                            scaled_testX, testY)

    print('best lambda : ', best_lambda)

    vis_mse(lambdas, MSE_set, best_lambda, best_MSE)

    cr_theta_0, cr_theta_1 = ridgeClosed_get_theta(scaled_trainX, trainY, best_lambda)
    cr_pred_train = ridgeClosed_predict(scaled_trainX, cr_theta_0, cr_theta_1)
    train_MSE_ridgeClosed = mean_squared_error(trainY, cr_pred_train)
    print('train MSE by closed Ridge : ', train_MSE_ridgeClosed)

    cr_pred_test = ridgeClosed_predict(scaled_testX, cr_theta_0, cr_theta_1)
    test_MSE_ridgeClosed = mean_squared_error(testY, cr_pred_test)
    print('test MSE by closed Ridge : ', test_MSE_ridgeClosed)



