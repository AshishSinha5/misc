import numpy as np

class MLP:
    """
    Creates an instance of a Multi-layer Perceptron (MLP) neural network with single hidden layer
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.w1 = np.random.randn(self.input_size, self.hidden_size) # shape = (input_size, hidden_size)
        self.b1 = np.zeros((1, self.hidden_size)) # shape = (1, hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))


    def forward(self, X):
        """
        Forward pass through the network

        Args:
            X : The input matrix (batch_size, input_size).
        """
        assert X.shape[-1] == self.input_size
        self.z1 = np.matmul(X, self.w1) + self.b1 # shape = (batch_size, hidden_size)
        self.a1  = self._relu(self.z1)
        self.z2 = np.matmul(self.a1,  self.w2) + self.b2 # shape = (batch_size, output_size)
        self.a2 = self._softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, lr=0.01):
        """
        Backward pass through the network
        """
        m = X.shape[0]
        dz2 = self.a2 - y
        dw2 = np.matmul(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        da1 = np.matmul(dz2, self.w2.T)
        dz1 = np.where(self.z1 > 0, da1, 0)
        dw1 = np.matmul(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # update the weights and biases
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2


    def _relu(self, X):
        """
        Relu activation function, performs relu activation on each of the feature value of X

        Args:
            X : The input matrix (batch_size, feature)

        Returns:
            The output matrix after applying the relu activation function
        """
        return np.where(X > 0, X, 0)
    
    def _softmax(self, X):
        """
        Sigmoid activation function, performs softmax activation on each of the feature vector, given a batch of input X

        Args:
            X : The input matrix (batch_size, feature)

        Returns:
            The output matrix after applying the softmax activation function
        """
        exps = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)
    
    def cross_entropy(self, y_true, y_pred):
        """
        Compute the cross-entropy loss between the true target and the predicted target

        Args:
            y_true : The true target matrix (batch_size, output_size)
            y_pred : The predicted target matrix (batch_size, output_size)

        Returns:
            The cross-entropy loss
        """
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    def accuracy(self, y_true, y_pred):
        """
        Compute the accuracy of the model

        Args:
            y_true : The true target matrix (batch_size, output_size)
            y_pred : The predicted target matrix (batch_size, output_size)

        Returns:
            The accuracy of the model
        """
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

    def train(self, X, y, epochs=100, lr=0.01, batch_size=32, train_val_split=0.2):
        """
        Train the model on the given dataset

        Args:
            X : The input matrix (batch_size, input_size)
            y : The target matrix (batch_size, output_size)
            epochs : The number of epochs to train the model
            lr : The learning rate
        """
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=train_val_split, random_state=42)
        for i in range(epochs):
            for j in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[j:j+batch_size]
                y_batch = y_train[j:j+batch_size]

                y_pred = self.forward(X_batch)
                loss = self.cross_entropy(y_batch, y_pred)
                history["train_loss"].append(loss)
                history["train_accuracy"].append(self.accuracy(y_batch, y_pred))

                self.backward(X_batch, y_batch, lr)

            # Evaluate the model on the validation set
            y_val_pred = self.forward(X_val)
            val_loss = self.cross_entropy(y_val, y_val_pred)
            val_acc = self.accuracy(y_val, y_val_pred)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            print(f"Epoch: {i+1}, Loss: {loss}, Accuracy: {val_acc}")

        return history

    def predict(self, X):
        """
        Predict the target values for the given input

        Args:
            X : The input matrix (batch_size, input_size)

        Returns:
            The predicted target matrix (batch_size, output_size)
        """
        return self.forward(X)
    

# load the mnist dataset from sklearn package
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split    

# Load the mnist dataset
print("Loading the MNIST dataset...")
mnist = fetch_openml('mnist_784', as_frame=False, parser='liac-arff')
print("MNIST dataset loaded successfully!")
X, y = mnist["data"], mnist["target"]
y = y.astype(int)

# one-hot encode the target variable
y = np.eye(10)[y]

# Normalize the input data
X = X / 255.0

model = MLP(input_size=784, hidden_size=128, output_size=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

history = model.train(X_train, y_train, epochs=10, lr=0.01, batch_size=32)

# Plot the training loss
import matplotlib.pyplot as plt

plt.plot(history["train_loss"], label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Train Loss")
plt.legend()
plt.show()

# Plot the validation loss
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.legend()
plt.show()



# Evaluate the model
y_pred = model.predict(X_test)
acc = model.accuracy(y_test, y_pred)

print(f"Test Accuracy: {acc}")


