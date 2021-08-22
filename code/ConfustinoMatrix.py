
import numpy as np


def Confusion_Matrix(target, pred):
    confu_mat = np.zeros((2, 2))

    for i in range(len(target)):
        if pred[i] == target[i]:
            if pred[i] == 0:
                confu_mat[1][1] += 1
            elif pred[i] == 1:
                confu_mat[0][0] += 1
        else:
            if pred[i] == 0 and target[i] == 1:
                confu_mat[0][1] += 1
            elif pred[i] == 1 and target[i] == 0:
                confu_mat[1][0] += 1

    return confu_mat


def precision(confu_mat):
    precision = 0

    TP = confu_mat[0][0]
    FP = confu_mat[1][0]

    precision = TP / (TP + FP)

    return precision


def recall(confu_mat):
    recall = 0

    TP = confu_mat[0][0]
    FN = confu_mat[0][1]

    recall = TP / (TP + FN)

    return recall


def f1(confu_mat):
    f1_score = 0

    TP = confu_mat[0][0]
    FN = confu_mat[0][1]
    FP = confu_mat[1][0]

    f1_score = TP / (TP + (FN + FP) / 2)

    return f1_score


def roc_auc(y_true, y_probs):
    desc_score_indices = np.argsort(y_probs)[::-1]
    y_true = y_true[desc_score_indices]
    threshold_idxs = np.arange(y_true.size)

    tps = np.cumsum(y_true)
    fps = 1 + np.arange(len(y_true)) - tps
    fpr = fps / float(fps[-1])
    tpr = tps / float(tps[-1])

    idx = np.argsort(fpr)
    fpr = fpr[idx]
    tpr = tpr[idx]

    area = np.trapz(tpr, fpr)

    return area, fpr, tpr

