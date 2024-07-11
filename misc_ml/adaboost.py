import numpy as np
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators = 50, learning_rate = 1.0):
        self.n_estimators = n_estimators # number of weak learners
        self.learning_rate = learning_rate # adjust the contribution of each weak learner
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X, y):
        n_samples = len(X)
        sample_weights = np.ones(n_samples) / n_samples
        print("Training the model")
        for i in tqdm(range(self.n_estimators)):

            # train the week learner, stump with max_depth=1
            estimator = DecisionTreeClassifier(max_depth=1)
            estimator.fit(X, y, sample_weight=sample_weights)

            # make the predictions on this weak learner
            y_pred = estimator.predict(X)

            # find the misclassified samples
            incorrect = y_pred != y
            weigthed_error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)

            # compute the predictor weight
            estimator_weight = self.learning_rate * 0.5 * np.log((1 - weigthed_error) / max(weigthed_error, 1e-10))

            # update the sample weights
            sample_weights = sample_weights*np.exp(-estimator_weight*y*y_pred)

            # normalize the sample weights
            sample_weights /= np.sum(sample_weights)

            # save the estimator and its weight
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)

    def predict(self, X):
        n_samples = len(X)
        y_pred = np.zeros(n_samples)

        for estimator, weight in zip(self.estimators, self.estimator_weights):
            y_pred += weight * estimator.predict(X)

        return np.sign(y_pred)
    

# load breast cancer dataset from sklearn
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target
y = 2*y - 1

# split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print('Number of training samples:', len(X_train))
print('Number of testing samples:', len(X_test))

print("Proportion of positive samples in training set:", sum(y_train == 1)/len(y_train))
print("Proportion of positive samples in testing set:", sum(y_test == 1)/len(y_test))


# train the model
model = AdaBoost(n_estimators=50)

model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# calculate the accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)