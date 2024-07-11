import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle, resample


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2, max_features='sqrt', bootstrap=True):
        self.n_estimators = n_estimators # number of trees
        self.max_depth = max_depth # maximum depth of each tree in the forest, prevent overfitting
        self.min_samples_split = min_samples_split # minimum number of samples required to split an internal node
        self.max_features = max_features # maximum number of features to consider for the best split
        self.bootstrap = bootstrap  
        self.estimators = []


    def fit(self, X, y):
        self.estimators = []
        n_samples, n_features = X.shape

        # max features to consider for each tree
        if self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            self.max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * n_features)
        else:
            self.max_features = n_features

        for _ in range(self.n_estimators):
            # create a bootstrap sample
            if self.bootstrap:
                X_boot, y_boot = resample(X, y)
            else:
                X_boot, y_boot = X, y

            # create a decision tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth, 
                                          min_samples_split=self.min_samples_split, 
                                          max_features=self.max_features)
            

            tree.fit(X_boot, y_boot)

            self.estimators.append(tree)

    
    def predict(self, X):

        n_samples = X.shape[0]
        n_estimators = len(self.estimators)
        predictions = np.zeros((n_estimators, n_samples), dtype=int)
        for i, tree in enumerate(self.estimators):
            predictions[i] = tree.predict(X)

        # return the mode of the predictions
        # np.bincount returns the count of each element in the array
        # np.argmax returns the index of the maximum element
        # predictions is a 2D array of shape (n_estimators, n_samples)
        return np.array([np.argmax(np.bincount(predictions[:, i])) for i in range(n_samples)])
    


# load breast cancer dataset from sklearn
data = load_breast_cancer()

X = data.data
y = data.target

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Number of training samples:', len(X_train))
print('Number of testing samples:', len(X_test))

rf = RandomForest(n_estimators=100, max_depth=5, min_samples_split=2, max_features='sqrt', bootstrap=True)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)