import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

from sklearn import metrics, pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import xgboost as xgb

from nltk import word_tokenize

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

print('Loading Embeddings')
with open('data/embeddings.pkl', 'rb') as f:
    embeddings_index = pickle.load(f)


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


mll_scorer = metrics.make_scorer(loss_metric, greater_is_better=False, needs_proba=True)


def sent2vec(words, normalized=True):
    words = word_tokenize(words)
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    if normalized:
        v = M.sum(axis=0)
        if type(v) != np.ndarray:
            return np.zeros(300)
        return v / np.sqrt((v ** 2).sum())
    return M


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


def glove_feats(X_train, X_valid, X_test):
    x_train_glv = np.array([sent2vec(x, normalized=False) for x in tqdm(X_train)])
    x_valid_glv = np.array([sent2vec(x, normalized=False) for x in tqdm(X_valid)])
    x_test_glv = np.array([sent2vec(x, normalized=False) for x in tqdm(X_test)])
    return x_train_glv, x_valid_glv, x_test_glv


def glove_feats_normalized(X_train, X_valid, X_test):
    x_train_glv = np.array([sent2vec(x) for x in tqdm(X_train)])
    x_valid_glv = np.array([sent2vec(x) for x in tqdm(X_valid)])
    x_test_glv = np.array([sent2vec(x) for x in tqdm(X_test)])
    return x_train_glv, x_valid_glv, x_test_glv


def logistic_model(xtrain, xvalid, xtest, y_train, y_valid):
    clf = LogisticRegression(C=1, max_iter=1000)
    clf.fit(xtrain, y_train)
    predictions = clf.predict_proba(xvalid)
    pred = clf.predict_proba(xtest)
    loss = multiclass_logloss(predictions, y_valid)
    score = clf.score(xvalid, y_valid)
    return pred, loss, score


def nb_model(xtrain, xvalid, xtest, y_train, y_valid):
    clf = MultinomialNB()
    clf.fit(xtrain, y_train)
    predictions = clf.predict_proba(xvalid)
    pred = clf.predict_proba(xtest)
    loss = multiclass_logloss(predictions, y_valid)
    score = clf.score(xvalid, y_valid)
    return pred, loss, score


def nb_grid_search(xtrain, xvalid, xtest, y_train, y_valid):
    nb_model = MultinomialNB()
    clf = pipeline.Pipeline([('nb', nb_model)])
    param_grid = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                         verbose=0, n_jobs=-1, refit=True, cv=2)

    model.fit(xtrain, y_train)
    predictions = model.predict_proba(xvalid)
    pred = model.predict_proba(xtest)
    loss = multiclass_logloss(predictions, y_valid)
    score = accuracy_score(y_valid, model.predict(xvalid))
    return pred, loss, score


def xgb_glove_model(xtrain, xvalid, xtest, y_train, y_valid):
    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                            subsample=0.8, nthread=10, learning_rate=0.1, silent=False, verbosity=0)

    clf.fit(xtrain, y_train)
    predictions = clf.predict_proba(xvalid)
    pred = clf.predict_proba(xtest)
    loss = multiclass_logloss(predictions, y_valid)
    score = clf.score(xvalid, y_valid)
    return pred, loss, score


def xgb_grid_search(xtrain, xvalid, xtest, y_train, y_test):
    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, subsample=0.8, nthread=10,
                            learning_rate=0.1, silent=False, verbosity=0, use_label_encoder=False)

    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'n_estimators': [180, 190, 200],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }

    model = RandomizedSearchCV(clf, param_distributions=params, scoring=mll_scorer, n_jobs=-1,
                               verbose=1, cv=2)

    model.fit(xtrain, y_train)
    predictions = model.predict_proba(xvalid)
    pred = model.predict_proba(xtest)
    loss = multiclass_logloss(predictions, y_valid)
    score = accuracy_score(y_valid, model.predict(xvalid))
    return pred, score, loss


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
    xtrain_glv, xvalid_glv, xtest_glv = glove_feats(X_train, X_valid, X_test)
    xtrain_glv_norm, xvalid_glv_norm, xtest_glv_norm = glove_feats_normalized(X_train, X_valid, X_test)

    lr_tfv, lr_tfv_loss, lr_tfv_score = logistic_model(xtrain_tfv, xvalid_tfv, xtest_tfv, y_train, y_valid)
    print('Logistic_regression with TFIDF : {}\tacc {}'.format(lr_tfv_loss, lr_tfv_score))
    save_op(lr_tfv, ids, 'lr', 'tfv')

    lr_ctv, lr_ctv_loss, lr_ctv_score = logistic_model(xtrain_ctv, xvalid_ctv, xtest_ctv, y_train, y_valid)
    print('Logistic_regression with Count Vectorizer : {}\tacc {}'.format(lr_ctv_loss, lr_ctv_score))
    save_op(lr_ctv, ids, 'lr', 'ctv')

    nb_tfv, nb_tfv_loss, nb_tfv_score = nb_model(xtrain_tfv, xvalid_tfv, xtest_tfv, y_train, y_valid)
    print('NB with TFIDF : {}\tacc {}'.format(nb_tfv_loss, nb_tfv_score))
    save_op(nb_tfv, ids, 'nb', 'tfv')

    nb_ctv, nb_ctv_loss, nb_ctv_score = nb_model(xtrain_ctv, xvalid_ctv, xtest_ctv, y_train, y_valid)
    print('NB with Count Vectorizer : {}\t{}'.format(nb_ctv_loss, nb_ctv_score))
    save_op(nb_ctv, ids, 'nb', 'ctv')

    nb_tuned_tfv, nb_tuned_tfv_loss, nb_tuned_tfv_score = nb_grid_search(xtrain_tfv, xvalid_tfv, xtest_tfv, y_train,
                                                                         y_valid)
    print('NB tuned with TFIDF : {}\tacc {}'.format(nb_tuned_tfv_loss, nb_tuned_tfv_score))
    save_op(nb_tuned_tfv, ids, 'nb_tuned', 'tfv')

    nb_tuned_ctv, nb_tuned_ctv_loss, nb_tuned_ctv_score = nb_grid_search(xtrain_ctv, xvalid_ctv, xtest_ctv, y_train,
                                                                         y_valid)
    print('NB tuned with Count Vectorizer : {}\tacc {}'.format(nb_tuned_ctv_loss, nb_tuned_ctv_score))
    save_op(nb_tuned_ctv, ids, 'nb_tuned', 'ctv')

    """# word embedding model
    xgb_glv, xgb_loss, xgb_score = xgb_glove_model(xtrain_glv_norm, xvalid_glv_norm, xtest_glv_norm, y_train, y_valid)
    print('XGBoost with Glove vectors : {}\tacc {}'.format(xgb_loss, xgb_score))
    save_op(xgb_glv, ids, 'xgb', 'glv')
    
    xgb_tuned_glv, xgb_tuned_loss, xgb_tuned_score = xgb_grid_search(xtrain_glv_norm, xvalid_glv_norm, xtest_glv_norm,
                                                                     y_train, y_valid)
    print('XGBoost tuned with Glove vectors : {}\tacc {}'.format(xgb_tuned_loss, xgb_tuned_score))
    save_op(xgb_tuned_glv, ids, 'xgb_tuned', 'glv')

    xgb_tuned_ctv, xgb_tuned_loss, xgb_tuned_score = xgb_grid_search(xtrain_ctv, xvalid_ctv, xtest_ctv,
                                                                     y_train, y_valid)
    print('XGBoost tuned with Glove vectors : {}\tacc {}'.format(xgb_tuned_loss, xgb_tuned_score))
    save_op(xgb_tuned_ctv, ids, 'xgb_tuned', 'ctv')"""
