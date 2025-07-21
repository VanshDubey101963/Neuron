import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from keras.api.models import Sequential
from keras.api.layers import Dense

class NeuralNetwork:
    def __init__(self, max_iterations=1000):
        self.W1, self.W2, self.W3 , self.B1 , self.B2 , self.B3 = None,None,None,None,None,None
        self.__learning_rate = 0.05
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

        self.W1 = np.random.rand(32,64) 
        self.B1 = np.zeros((32,1))  
        self.W2 = np.random.rand(32,32)
        self.B2 = np.zeros((32,1))
        self.W3 = np.random.rand(10,32)
        self.B3 = np.zeros((10,1))

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
        

    def fit(self,X_r,y):
        X_r = X_r.T.reshape(64,-1)
        onehot_y = self.__onehot(y)
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

def one_hot(y):
    one_hot_encoded = np.eye(10)[y]
    return one_hot_encoded.T

digits = load_digits()

X = digits.data / 16
Y = digits.target
Y = Y.astype(int)
nn = NeuralNetwork()
nn.fit(X,Y)
prediction = nn.predict(X[2].T.reshape(64,-1))
print(prediction)
print(one_hot(Y[2]))
print(one_hot(Y[2]).argmax())
print(prediction.argmax())