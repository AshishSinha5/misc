import random 
import math
import numpy as np
import matplotlib.pyplot as plt 

from sklearn import datasets

def to_one_hot(y):
    b = np.zeros((y.size, y.max() + 1))
    b[np.arange(y.size), y] = 1
    return b

class MultiClassLogistic:
    def __init__(self, num_class, learning_rate = 0.001, iteration = 1000, batch_size = None, log_every = 100):
        self.num_class = num_class
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.batch_size = batch_size
        self.log_every = log_every
        self.intercept = None 
        self.coefficients = None 
        self.loss = []

    def fit(self, X, y):
        """
        X = [num_samples, num_feat]
        y = [num_samples, num_class]  one hot encoded
        """
        X = (X - X.mean())/X.std()
        num_samples, num_feat = X.shape

        self.intercept = np.random.random((self.num_class, 1))
        self.coefficients = np.random.random((self.num_class, num_feat))
        self.loss = []

        for i in range(self.iteration):
            if self.batch_size:
                random_index = np.random.choice(num_samples, self.batch_size, replace=False)
                X_batch = X[random_index]
                y_batch = y[random_index]
            else:
                X_batch = X
                y_batch = y

            y_pred = self.predict(X_batch)  # [num_sample, num_class]
            self.loss.append(self.nll(y_batch, y_pred))
            
            error = y_pred - y_batch
            self.intercept -= self.learning_rate * np.sum(error, axis=0, keepdims=True).T
            self.coefficients -= self.learning_rate * (X_batch.T @ error).T

            if i>0 and i%self.log_every == 0:
                print(f"Loss at iteration {i} = {self.loss[-1]}")

    def predict(self, X):
        """
        X = [num_samples, num_feat]
        return [num_samples, num_class]
        """
        Z = X @ self.coefficients.T + self.intercept.T
        return self.softmax(Z)
    
    def softmax(self, Z):
        exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def nll(self, y_true, y_pred):
        """
        y_true = [num_sample, num_class]
        y_pred = [num_sample, num_class]
        """
        return -np.sum(y_true * np.log(y_pred + 1e-15))

X, y = datasets.load_iris(return_X_y = True)
y = to_one_hot(y)

print(X.shape)
print(y.shape)

model = MultiClassLogistic(num_class=len(y[0]), learning_rate=0.01, iteration=100000, log_every=10000)
model.fit(X, y)

plt.plot(model.loss)
plt.xlabel('Iteration')
plt.ylabel('Negative Log-Likelihood')
plt.title('Training Loss')
plt.show()