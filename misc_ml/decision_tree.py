import math
from collections import Counter


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth = 0):
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth) or len(X) == 0:
            return Counter(y).most_common(1)[0][0] # return the most common element (mejority class voting)
        

        # find the best feature and best gain for the feature split 
        best_feature, best_value, best_gain = self._find_best_split(X, y)

        left_X = [x for x in X if x[best_feature] <= best_value]
        left_y = [y[i] for i in range(len(y)) if X[i][best_feature] <= best_value]
        right_X = [x for x in X if x[best_feature] > best_value]
        right_y = [y[i] for i in range(len(y)) if X[i][best_feature] > best_value]

        tree = {
            "feature" : best_feature,
            "value" : best_value,
            "left" : self._build_tree(left_X, left_y),
            "right" : self._build_tree(right_X, right_y)
        }
        return tree
    
    def _find_best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_value = None

        current_entropy = self._entropy(y)

        # iterate through the X matrix
        # for all the features of the X 
        # find the value which gives the best information gain 

        for feature in range(len(X[0])):
            values = sorted(set([x[feature] for x in X]))

            for i in range(len(values) - 1):
                # find the mean value in current bucket 
                value = (values[i] + values[i+1])/2 

                left_y = [y[i] for i in range(len(y)) if X[i][feature] <= value]
                right_y = [y[i] for i in range(len(y)) if X[i][feature] > value]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                weighted_entropy = len(left_y)/len(y)*self._entropy(left_y) + len(right_y)/len(y)*self._entropy(right_y)


                info_gain = current_entropy - weighted_entropy

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature = feature
                    best_value = value

        return best_feature, best_value, best_gain
    

    def _entropy(self, y):
        counts = Counter(y)
        entropy = 0
        for count in counts.values():
            p = count/len(y)
            entropy -= p*math.log2(p)

        return entropy 
    
    def predict(self, X):
        return [self._predict_single(x, self.tree) for x in X]

    def _predict_single(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        
        if x[tree['feature']] <= tree['value']:
            return self._predict_single(x, tree['left'])

        return self._predict_single(x, tree['right'])
# Example usage
X = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [6.3, 3.3, 6.0, 2.5],
    [5.8, 2.7, 5.1, 1.9]
]

y = ['setosa', 'setosa', 'versicolor', 'versicolor', 'virginica', 'virginica']

dt = DecisionTree(max_depth=3)
dt.fit(X, y)

print("Decision Tree:")
print(dt.tree)


# Make predictions
new_flowers = [
    [5.0, 3.4, 1.5, 0.2],
    [6.7, 3.1, 4.7, 1.5],
    [6.3, 3.3, 6.0, 2.5],
]

predictions = dt.predict(new_flowers)
print("\nPredictions:")
for flower, prediction in zip(new_flowers, predictions):
    print(f"Flower {flower}: {prediction}")
