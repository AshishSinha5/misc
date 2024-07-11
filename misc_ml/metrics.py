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
    f1 = 2pr/(p+r)
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

        # true positive rate - proportion of positives that are correctly classified as positive
        tpr = tp/(tp + fn) if (tp + fn) > 0 else 0
        # false positive rate - proportion of negatives that are incorrectly classified as positive
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


def precisionAtK(y_true, y_scores, k):
    """
    calculate the precision at k
    Preicision at k is the proportion of relevant documents among the top k documents
    """
    assert len(y_true) == len(y_scores)
    assert k > 0

    y_pred = []
    top_k_scores = sorted(y_scores, reverse=True)[:k]
    for score in y_scores:
        if score in top_k_scores:
            y_pred.append(1)
        else:
            y_pred.append(0)
    # precision is the accuracy of the positive prediction
    # precision = tp/(tp + fp)
    return precision(y_true, y_pred)


def recallAtK(y_true, y_scores, k):
    """
    calculate the recall at k
    Recall at k is the proportion of relevant documents found in the top k documents
    """
    assert len(y_true) == len(y_scores)
    assert k > 0

    y_pred = []
    top_k_scores = sorted(y_scores, reverse=True)[:k]
    for score in y_scores:
        if score in top_k_scores:
            y_pred.append(1)
        else:
            y_pred.append(0)
    # recall is completness of the positive prediction
    # recall = tp/(tp + fn)
    return recall(y_true, y_pred) 


def f1AtK(y_true, y_scores, k):
    """
    calculate the f1 at k
    f1 at k is the harmonic mean of precision and recall at k
    """
    assert len(y_true) == len(y_scores)
    assert k > 0

    y_pred = []
    top_k_scores = sorted(y_scores, reverse=True)[:k]
    for score in y_scores:
        if score in top_k_scores:
            y_pred.append(1)
        else:
            y_pred.append(0)
    # f1 = 2pr/(p+r)
    return f1(y_true, y_pred)

def precisionAtRecall(y_true, y_scores, recall_level):
    """
    calculate the precision at a given recall level
    """
    assert len(y_true) == len(y_scores)
    assert 0 <= recall_level <= 1

    y_pred = []
    for score in y_scores:
        if score >= recall_level:
            y_pred.append(1)
        else:
            y_pred.append(0)
    # precision is the accuracy of the positive prediction
    # precision = tp/(tp + fp)
    return precision(y_true, y_pred)


# impliment precision recall curve
def pr(y_true, y_scores):
    """
    Precision Recall curve
    precision vs recall
    a good model will have a curve closer to (1, 1) 
    a random model will be a stright line passing through (0,0) and (1, 1)
    a bad model does worse than the random model and has curve passing below the random model line

    more closer the curve is to the point (1, 1), better the model is.
    """
    thresholds = sorted(set(y_scores), reverse=True)
    
    pr_points = []

    for threshold in thresholds:
        y_pred = [1*(score >= threshold) for score in y_scores]
        tp = sum(1 for t, p in zip(y_true, y_pred) if y_true == 1 and y_pred == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if y_true == 1 and y_pred == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if y_true == 0 and y_pred == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if y_true == 0 and y_pred == 0)

        # precision 
        precision = tp/(tp + fp) if (tp + fp) > 0 else 0
        # recall
        recall = tp/(tp + fn) if (tp + fn) > 0 else 0

        pr_points.append((recall, precision, threshold))

    return sorted(pr_points)




