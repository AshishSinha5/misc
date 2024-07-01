import math
import random
import matplotlib.pyplot as plt 
import numpy as np

from sklearn import datasets


class LogisticRegression:
    def __init__(self, learning_rate=0.001, iteration = 1000, batch_size = None):
        self.learning_rate = learning_rate
        self.iteration = iteration 
        self.batch_size = batch_size 
        self.intercept = None
        self.coefficients = None
        self.loss = [] 


    def fit(self, X, y):
        X = (X - X.mean())/X.std()
        y = np.expand_dims(y, -1)
        num_samples, num_feat = len(X), len(X[0])
        self.loss = []

        # intialize intercept 
        self.intercept = random.random()
        self.coefficients = [[random.random()] for _ in range(num_feat)]

        for i in range(self.iteration):
            if self.batch_size:
                random_index = random.sample(range(num_samples), k=self.batch_size)
                X_batch = X[random_index]
                y_batch = y[random_index]
            else:
                X_batch = X
                y_batch = y
        
            y_pred = self.predict(X_batch)
            self.loss.append(self.nll(y_batch, y_pred))

            # update rule
            # theta = theta - alpha*del(nll)/del(theta)
            self.intercept += self.learning_rate*(np.sum(y_batch - y_pred))
            self.coefficients += self.learning_rate*(np.matmul(X_batch.transpose(), (y_batch - y_pred)))



    def predict(self, X):
        theta_x = self.intercept + np.matmul(X, self.coefficients)
        return self.sigmoid(theta_x)
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def nll(self, y_true, y_pred):
        return -np.sum(y_true*np.log2(y_pred) + (1 - y_true)*np.log2(1 - y_pred))/len(y_pred)
    
X, y = datasets.load_breast_cancer(return_X_y=True)

model = LogisticRegression(learning_rate=0.0001, iteration=10000, batch_size=32)
model.fit(X, y)

print(f"Intercept = {model.intercept:.4f}")
print(f"Coeffitients = {[f'{coef[0]:.4f}' for coef in model.coefficients]}")

plt.plot(model.loss)
plt.show()
