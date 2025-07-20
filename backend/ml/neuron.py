import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

class NeuralNetwork:
    def __init__(self, max_iterations=10000):
        self.W1, self.W2, self.W3 , self.B1 , self.B2 , self.B3 = None,None,None,None,None,None
        self.__learning_rate = 0.01
        self.max_iterations = max_iterations

    def __softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __dsigmoid(self, z):
        return z * (1-z)
    
    def __dsoftmax(self, y_hat , y):
        return y_hat - y
    
    def initialize_params(self):

        self.W1 = np.random.rand(32,784) * 0.01
        self.B1 = np.random.rand(32,1) * 0.01
        self.W2 = np.random.rand(32,32) * 0.01
        self.B2 = np.random.rand(32,1) * 0.01
        self.W3 = np.random.rand(10,32) * 0.01
        self.B3 = np.random.rand(10,1) * 0.01

    def forward_prop(self,X):
        
        Z1 = self.W1 @ X + self.B1
        A1 = self.__sigmoid(Z1)

        Z2 = self.W2 @ A1 + self.B2
        A2 = self.__sigmoid(Z2)

        Z3 = self.W3 @ A2 + self.B3
        A3 = self.__softmax(Z3)

        mem = (Z1, A1, Z2, A2, Z3, A3)
        return A3, mem


    def backward_prop(self,X, Y, mem):

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

    def __onehot(self,y):
        one_hot_encoded = np.eye(10)[y]
        return one_hot_encoded.T
        

    def fit(self,X,y):
        onehot_y = self.__onehot(y)
        X_r = np.reshape(X, (784,1000))

        self.initialize_params()


        for i in range(1,self.max_iterations + 1):

            if i % 100 == 0 :
                    print(f"Epoch {i} complete")

            A3, mem = self.forward_prop(X_r)
            dW1, db1 , dW2, db2, dW3, db3 = self.backward_prop(X_r, onehot_y,mem)

            self.W1 -= self.__learning_rate * dW1
            self.B1 -= self.__learning_rate * db1
            self.W2 -= self.__learning_rate * dW2
            self.B2 -= self.__learning_rate * db2
            self.W3 -= self.__learning_rate * dW3
            self.B3 -= self.__learning_rate * db3

    
    def predict(self,X):
        A3 , mem = self.forward_prop(X)
        return A3

mnist = fetch_openml('mnist_784', as_frame=False)

X = mnist.data[0:1000]
Y = mnist.target[0:1000]
new_Y = Y.astype(int)
nn = NeuralNetwork()
nn.fit(X,new_Y)