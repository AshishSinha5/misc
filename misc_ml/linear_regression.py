import random 
import math 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class LinearRegrssion:
    def __init__(self, learning_rate = 0.001, iteration = 1000, batch_size = None):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.batch_size = batch_size
        self.intercept = None
        self.coefficients = None
        self.loss = []
    
    def fit(self, X, y):
        # X = (X - X.mean())/X.std()
        y = np.expand_dims(y, -1)
        num_samples, p = len(X), X.shape[1]
        self.loss = []
        """
        gradient update rule - 
        theta_j = theta_j + alpha*Sum(y - theta^Tx)x_j
        """
        # initial guess for theta
        self.intercept = np.random.random()
        self.coefficients = [[np.random.random()] for _ in range(p)]
        for i in range(self.iteration):
            if self.batch_size:
                random_indices = random.sample(range(len(X)), k=self.batch_size)
                x_batch = X[random_indices]
                y_batch = y[random_indices]
            else:
                x_batch = X
                y_batch = y

            y_pred = self.predict(x_batch)
            self.loss.append(self.mse(y_batch, y_pred))

            """
            x_batch - n,p
            y_batch - y_pred - n,1
            p,1
            """
            self.intercept += self.learning_rate*(np.sum(y_batch - y_pred))
            self.coefficients += self.learning_rate*(np.matmul(x_batch.transpose(), (y_batch - y_pred)))

    def predict(self, X):
        """
        X = n, p
        coeff = p, 1
        """
        assert len(X.shape) == 2 # (num_samples, num_feat)
        return self.intercept + np.matmul(X, self.coefficients)

    def mse(self, y_true, y_pred):
        return sum([y - y_hat for y, y_hat in zip(y_true, y_pred)])/len(y_pred)
    
    def r2(self, X, y):
        """
        r2 = 1 - rss/tss
        tss = Sum (y - y_mean)^2
        rss = Sum (y - y_i)^2
        """
        y_pred = self.predict(X)
        y_mean = np.mean(y)

        tss = np.sum((y - y_mean)**2)
        rss = np.sum((y - y_pred)**2)

        return 1 - rss/tss
    
    def __repr__(self) -> str:
        s = f"Intercept = {self.intercept:.4f}"
        
        s += f"Coeffitients - \n"
        for coeffs in np.squeeze(self.coefficients):
            s += f"{coeffs:.4f} \t"
        
        return s


# Example usage
X, y = datasets.load_diabetes(return_X_y=True)
X = X[:, :5]
model = LinearRegrssion(learning_rate=0.001, iteration=1000, batch_size=64)
model.fit(X, y)
plt.plot(model.loss)
plt.show()
print(model)



