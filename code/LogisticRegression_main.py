

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score

from LogisticRegression_model import *
from ConfustinoMatrix import *
from utils import *


def dataProcessing():
    df = pd.read_csv('./data2.csv').dropna()

    y = df.Outcome

    X = df.drop(['Outcome'], axis=1).astype('float64')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=False)

    standardizer = StandardScaler()
    scaled_X_train = standardizer.fit_transform(X_train)
    scaled_X_test = standardizer.transform(X_test)

    return X_train, y_train, X_test, y_test, scaled_X_train, scaled_X_test, X.columns


if __name__ == "__main__":
    trainX, trainY, testX, testY, scaled_trainX, scaled_testX, colX = dataProcessing()

    logistic = l2_logistic(alpha=0.001, lamb=0)
    logistic.fit(scaled_trainX, trainY)
    tr_probs, tr_pred = logistic.predict(scaled_trainX)
    log_probs, pred = logistic.predict(scaled_testX)

    print('\n### Check Accuracy at lambda 0 ###')
    print('Logistic Training Accuracy : ' + str(np.mean(tr_pred == trainY)))
    print('Logistic Test Accuracy: ', str(np.mean(pred == testY)))

    l2_logistic_reg = l2_logistic(alpha=0.001, lamb=0.5)
    l2_logistic_reg.fit(scaled_trainX, trainY)
    tr_probs, tr_reg_pred = l2_logistic_reg.predict(scaled_trainX)
    probs, reg_pred = l2_logistic_reg.predict(scaled_testX)

    print('\n### Check Accuracy at L2 logistic ###')
    print('L2_logistic Training Accuracy : ' + str(np.mean(tr_reg_pred == trainY)))
    print('L2_logistic Test Accuracy: ', str(np.mean(reg_pred == testY)))

    l1_logistic = SGDClassifier(penalty='l1', loss='log',alpha=0.0001,random_state=4)
    l1_logistic.fit(scaled_trainX, trainY)

    print('\n### Check Accuracy at L1 logistic ###')
    print ('L1_logistic Training Accuracy : ' + str(np.mean(l1_logistic.predict(scaled_trainX) == trainY)))
    print ('L1_logistic Test Accuracy: ', str(np.mean(l1_logistic.predict(scaled_testX) == testY)))

    # Performance Measure
    Y_true = testY
    Y_true = np.array(Y_true)
    l2_probs, Y_l2_logistic_pred = l2_logistic_reg.predict(scaled_testX)

    confu_mat = Confusion_Matrix(Y_true, Y_l2_logistic_pred)
    l2_logistic_precisions = precision(confu_mat)
    l2_logistic_recalls = recall(confu_mat)
    l2_logistic_f1 = f1(confu_mat)
    print('\n### Evaluations for logistic regression with L2-norm regularization ###')
    print('precision :', l2_logistic_precisions)
    print('recall :', l2_logistic_recalls)
    print('F1 score :', l2_logistic_f1)

    Y_true = testY
    Y_true = np.array(Y_true)
    Y_l1_logistic_pred = l1_logistic.predict(scaled_testX)
    l1_probs = l1_logistic.predict_proba(scaled_testX)
    l1_probs = l1_probs[:, 1]

    confu_mat = Confusion_Matrix(Y_true, Y_l1_logistic_pred)
    l1_logistic_precisions = precision(confu_mat)
    l1_logistic_recalls = recall(confu_mat)
    l1_logistic_f1 = f1(confu_mat)

    print('\n### Evaluations for logistic regression with L1-norm regularization ###')
    print('precision :', l1_logistic_precisions)
    print('recall :', l1_logistic_recalls)
    print('F1 score :', l1_logistic_f1)

    precisions, recalls, thresholds = precision_recall_curve(Y_true, l1_probs)
    score = average_precision_score(Y_true, l1_probs)
    plotPR(score, recalls, precisions)

    precisions, recalls, thresholds = precision_recall_curve(Y_true, l2_probs)
    score = average_precision_score(Y_true, l2_probs)
    plotPR(score, recalls, precisions)

    l2_area, l2_fpr, l2_tpr = roc_auc(Y_true, l2_probs)
    l1_area, l1_fpr, l1_tpr = roc_auc(Y_true, l1_probs)

    print('\n### Check AUROC ###')
    print('AUROC of L2-norm regression classifier :', l2_area)
    print('AUROC of L1-norm regression classifier :', l1_area)

    plotROC(l2_area, l2_fpr, l2_tpr)

    plotROC(l1_area,l1_fpr,l1_tpr)
