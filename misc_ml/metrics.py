import math 
import numpy as np


def precision(y_true, y_pred):
    """
    Measured the accuracy of the positive predictions
    Out of all predicted positives, what proportions are actually correct
    assume binary classification
    precision = TP/(TP + FP)
    """
    assert len(y_true) == len(y_pred)
    # true positive - instance where y_true == y_pred == 1
    tp = sum([yt*yp for yt, yp in zip(y_true, y_pred)])
    precision = tp/sum(y_pred)
    return precision if sum(y_pred) else 0.0


def recall(y_true, y_pred):
    """
    Measure the completness of the positive prediction
    What proportion of positives are we able to detect
    assume binary classification 
    recall = tp/(tp + fn)
    """
    assert len(y_true) == len(y_pred)
    tp = sum([yt*yp for yt, yp in zip(y_true, y_pred)])
    recall = tp/sum(y_true)
    return recall if sum(y_true) else 0.0


def f1(y_true, y_pred):
    """
    f1 = pr/(p+r)
    """

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*p*r/(p+r)

def f_beta(y_true, y_pred, beta):
    """
    f_beta = (1 + beta**2)pr/(beta**2*p + r)
    """

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return (1 + beta**2)*p*r/((beta**2)*p + r)

def roc(y_true, y_proba):
    """
    ROC curve
    TPR vs FPR 
    a good model will have a curve closer to (0, 1) 
    a random model will be a stright line passing through (0,0) and (1, 1)
    a bad model does worse than the random model and has curve passing below the random model line

    more closer the curve is to the point (0, 1), better the model is.
    """
    thresholds = sorted(set(y_proba), reverse=True)
    
    roc_points = []

    for threshold in thresholds:
        y_pred = [1*(proba >= threshold) for proba in y_proba]
        tp = sum(1 for t, p in zip(y_true, y_pred) if y_true == 1 and y_pred == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if y_true == 1 and y_pred == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if y_true == 0 and y_pred == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if y_true == 0 and y_pred == 0)

        tpr = tp/(tp + fn) if tp + fn > 0 else 0
        fpr = fp/(fp + tn) if (fp + tn) > 0 else 0

        roc_points.append((fpr, tpr, threshold))

    return sorted(roc_points)

def auc(y_true, y_scores):
    """
    calculate the area under roc curve using the trapezoid rule
    """
    roc_points = roc(y_true, y_scores)
    auc = 0
    for i in range(1, len(roc_points)):
        x1, y1, _ = roc_points[i-1]
        x2, y2, _ = roc_points[i]
        auc += (x2-x1)*(y2+y1)/2

    return auc



