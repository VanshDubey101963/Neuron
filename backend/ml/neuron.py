import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    def __init__(self, max_iterations=250):
        self.W1, self.W2, self.W3, self.B1, self.B2, self.B3 = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self.__learning_rate = 0.8
        self.max_iterations = max_iterations

    def __softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __dsigmoid(self, z):
        return z * (1 - z)

    def __dsoftmax(self, y_hat, y):
        return y_hat - y

    def __cross_entropy(self, Y, A3):
        m = Y.shape[1]
        epsilon = 1e-8
        loss = -np.sum(Y * np.log(A3 + epsilon)) / m
        return loss

    def initialize_params(self):
        self.W1 = np.random.randn(32, 64) * np.sqrt(1 / 64)
        self.B1 = np.zeros((32, 1))
        self.W2 = np.random.randn(32, 32) * np.sqrt(1 / 32)
        self.B2 = np.zeros((32, 1))
        self.W3 = np.random.randn(10, 32) * np.sqrt(1 / 32)
        self.B3 = np.zeros((10, 1))

    def forward_prop(self, X):

        Z1 = self.W1 @ X + self.B1
        A1 = self.__sigmoid(Z1)

        Z2 = self.W2 @ A1 + self.B2
        A2 = self.__sigmoid(Z2)

        Z3 = self.W3 @ A2 + self.B3
        A3 = self.__softmax(Z3)

        mem = (Z1, A1, Z2, A2, Z3, A3)

        return A3, mem

    def backward_prop(self, X, Y, mem):

        m = X.shape[1]
        Z1, A1, Z2, A2, Z3, A3 = mem

        dZ3 = self.__dsoftmax(A3, Y)
        dW3 = dZ3 @ A2.T / m
        db3 = np.sum(dZ3, axis=1, keepdims=True) / m
        dA2 = self.W3.T @ dZ3
        dZ2 = dA2 * self.__dsigmoid(A2)
        dW2 = dZ2 @ A1.T / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * self.__dsigmoid(A1)
        dW1 = dZ1 @ X.T / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        return (dW1, db1, dW2, db2, dW3, db3)

    def __onehot(self, y):
        one_hot_encoded = np.eye(10)[y]
        return one_hot_encoded.T

    def fit(self, X_r, y):
        X_r = X_r.T
        onehot_y = self.__onehot(y)
        self.initialize_params()

        for i in range(1, self.max_iterations + 1):

            A3, mem = self.forward_prop(X_r)
            dW1, db1, dW2, db2, dW3, db3 = self.backward_prop(X_r, onehot_y, mem)

            if i % 100 == 0:
                print(
                    f"Epoch {i} computed loss: {self.__cross_entropy(onehot_y,A3):2f}"
                )

            self.W1 -= self.__learning_rate * dW1
            self.B1 -= self.__learning_rate * db1
            self.W2 -= self.__learning_rate * dW2
            self.B2 -= self.__learning_rate * db2
            self.W3 -= self.__learning_rate * dW3
            self.B3 -= self.__learning_rate * db3

    def predict(self, X):
        A3, mem = self.forward_prop(X)
        return A3


def one_hot(y):
    one_hot_encoded = np.eye(10)[y]
    return one_hot_encoded.T


digits = load_digits()

X = digits.data / 16
Y = digits.target
Y = Y.astype(int)
nn = NeuralNetwork()
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
nn.fit(train_X, train_Y)
preds = nn.predict(test_X.T)
pred_labels = np.argmax(preds, axis=0)
accuracy = np.mean(pred_labels == test_Y)
print(f"Training Accuracy: {accuracy * 100:.2f}%")
