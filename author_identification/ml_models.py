import argparse
import time
import numpy as np
import pandas as pd

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from main import preprocess, multiclass_logloss

author_code = {
    'EAP': 'Edgar Allan Poe',
    'HPL': 'HP Lovecraft',
    'MWS': 'Mary Shelley'
}

label_code = {
    'EAP': 0,
    'HPL': 1,
    'MWS': 2
}

def loss_metric(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


def get_train_valid_data(train_file_path, test_file_path, valid_ratio):
    df = pd.read_csv(train_file_path)
    test = pd.read_csv(test_file_path)
    X_test = test['text']
    ids = test['id']
    X = df['text']
    X = list(map(lambda x: preprocess(x), X))  # preprocess - lower case lemmatize remove stop words
    X_test = list(map(lambda x: preprocess(x), X_test))
    y = df['author']
    y = np.array(list(map(lambda x: label_code[x], y)))
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_ratio, stratify=y)
    return X_train, X_valid, X_test, y_train, y_valid, ids


"""
tfidf
count_features
logistic_regression
naive_bayes
svm
xgboost
grid serach
"""


def tfidf_features(X_train, X_valid, X_test):
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)

    tfv.fit(list(X_train))
    X_train_tfv = tfv.transform(X_train)
    X_valid_tfv = tfv.transform(X_valid)
    X_test_tfv = tfv.transform(X_test)

    return X_train_tfv, X_valid_tfv, X_test_tfv


def freq_count_features(X_train, X_valid, X_test):
    ctv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3))

    ctv.fit(list(X_train))
    X_train_ctv = ctv.transform(X_train)
    X_valid_ctv = ctv.transform(X_valid)
    X_test_ctv = ctv.transform(X_test)

    return X_train_ctv, X_valid_ctv, X_test_ctv


def logistic_model(xtrain_tfv, xvalid_tfv, xtrain_ctv, xvalid_ctv, xtest_tfv, xtest_ctv, y_train, y_valid):
    clf = LogisticRegression(C=1, max_iter=1000)
    # xtrain_tfv, xvalid_tfv = tfidf_features(X_train, X_valid)
    clf.fit(xtrain_tfv, y_train)
    predictions_tfidf = clf.predict_proba(xvalid_tfv)
    tfv_pred = clf.predict_proba(xtest_tfv)
    tfidf_loss = multiclass_logloss(predictions_tfidf, y_valid)

    print("Logistic Model with TFIDF features : %0.3f " % tfidf_loss)

    # xtrain_ctv, xvalid_ctv = freq_count_features(X_train, X_valid)
    clf.fit(xtrain_ctv, y_train)
    predictions_ctv = clf.predict_proba(xvalid_ctv)
    ctv_loss = multiclass_logloss(predictions_ctv, y_valid)
    ctv_pred = clf.predict_proba(xtest_ctv)

    print("Logistic Model with Count Vectorizer features : %0.3f " % ctv_loss)

    return tfv_pred, ctv_pred


def nb_model(xtrain_tfv, xvalid_tfv, xtrain_ctv, xvalid_ctv, xtest_tfv, xtest_ctv, y_train, y_valid):
    clf = MultinomialNB()
    # xtrain_tfv, xvalid_tfv = tfidf_features(X_train, X_valid)
    clf.fit(xtrain_tfv, y_train)
    predictions_tfidf = clf.predict_proba(xvalid_tfv)
    tfv_pred = clf.predict_proba(xtest_tfv)
    tfidf_loss = multiclass_logloss(predictions_tfidf, y_valid)

    print("Naive Bayes model with TFIDF features : %0.3f " % tfidf_loss)

    # xtrain_ctv, xvalid_ctv = freq_count_features(X_train, X_valid)
    clf.fit(xtrain_ctv, y_train)
    predictions_ctv = clf.predict_proba(xvalid_ctv)
    ctv_loss = multiclass_logloss(predictions_ctv, y_valid)
    ctv_pred = clf.predict_proba(xtest_ctv)
    print("Naive Bayes model with Count Vectorizer features : %0.3f " % ctv_loss)

    return tfv_pred, ctv_pred


def nb_grid_search(xtrain_tfv, xvalid_tfv, xtrain_ctv, xvalid_ctv, xtest_tfv, xtest_ctv, y_train, y_valid):
    mll_scorer = metrics.make_scorer(loss_metric, greater_is_better=False, needs_proba=True)
    nb_model = MultinomialNB()
    clf = pipeline.Pipeline([('nb', nb_model)])
    param_grid = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                         verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

    model.fit(xtrain_tfv, y_train)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    predictions_tfidf = model.predict_proba(xvalid_tfv)
    tfv_pred = model.predict_proba(xtest_tfv)
    tfidf_loss = multiclass_logloss(predictions_tfidf, y_valid)

    print("Naive Bayes tuned model with TFIDF features : %0.3f " % tfidf_loss)

    model.fit(xtrain_ctv, y_train)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    predictions_ctv = model.predict_proba(xvalid_ctv)
    ctv_pred = model.predict_proba(xtest_ctv)
    ctv_loss = multiclass_logloss(predictions_ctv, y_valid)
    print("Naive Bayes model tuned with Count Vectorizer features : %0.3f " % ctv_loss)

    return tfv_pred, ctv_pred


def save_op(predictions, ids, model, feat):
    predictions = np.array(predictions)
    eap, hpl, mws = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    data = {
        'id': ids,
        'EAP': eap,
        'HPL': hpl,
        'MWS': mws
    }
    df = pd.DataFrame(data=data)
    df.to_csv('outputs/{}_{}.csv'.format(model, feat), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', type=str, default='data/train.csv', help='train_file_path')
    parser.add_argument('--test_file_path', type=str, default='data/test.csv', help='test_file_path')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='proportion of validation samples')

    args = parser.parse_args()

    train_file_path = args.train_file_path
    test_file_path = args.test_file_path
    valid_ratio = args.valid_ratio

    print('Data preprocessing')
    X_train, X_valid, X_test, y_train, y_valid, ids = get_train_valid_data(train_file_path, test_file_path, valid_ratio)
    print('Getting Features')
    xtrain_tfv, xvalid_tfv, xtest_tfv = tfidf_features(X_train, X_valid, X_test)
    xtrain_ctv, xvalid_ctv, xtest_ctv = freq_count_features(X_train, X_valid, X_test)

    lr_tfv, lr_ctv = logistic_model(xtrain_tfv, xvalid_tfv, xtrain_ctv, xvalid_ctv, xtest_tfv, xtest_ctv, y_train,
                                    y_valid)
    save_op(lr_tfv, ids, 'lr', 'tfv')
    save_op(lr_ctv, ids, 'lr', 'ctv')

    nb_tfv, nb_ctv = nb_model(xtrain_tfv, xvalid_tfv, xtrain_ctv, xvalid_ctv, xtest_tfv, xtest_ctv, y_train, y_valid)
    save_op(nb_tfv, ids, 'nb', 'tfv')
    save_op(nb_ctv, ids, 'nb', 'ctv')

    nb_tuned_tfv, nb_tuned_ctv = nb_grid_search(xtrain_tfv, xvalid_tfv, xtrain_ctv, xvalid_ctv, xtest_tfv, xtest_ctv,
                                                y_train, y_valid)

    save_op(nb_tuned_ctv, ids, 'nb_tuned', 'tfv')
    save_op(nb_tuned_tfv, ids, 'nb_tuned', 'ctv')


