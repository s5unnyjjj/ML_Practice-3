
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LassoRegression_model import *
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

    lambdas = 10 ** np.linspace(3, -1, 50)
    lasso = Lasso(max_iter=10000)
    coefs_lasso = []

    for a in lambdas:
        lasso.set_params(alpha=a)
        lasso.fit(scaled_trainX, trainY)
        coefs_lasso.append(lasso.coef_)

    vis_coef(lambdas,coefs_lasso, method='Lasso')

    lambdas = 10 ** np.linspace(1, -1, 20)
    MSE_set_lasso, best_MSE_lasso, best_lambda_lasso, test_MSE_lasso, lasso = crossValidation_Lasso(lambdas, 10,
                                                                                                    scaled_trainX,
                                                                                                    trainY,
                                                                                                    scaled_testX,
                                                                                                    testY)

    print('best_lambda : ', best_lambda_lasso)

    vis_mse(lambdas,MSE_set_lasso,best_lambda_lasso,best_MSE_lasso)