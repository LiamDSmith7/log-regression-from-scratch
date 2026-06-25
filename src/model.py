import csv
import math
import json
import pickle
from pathlib import Path
import numpy as np 

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        )
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            loss = self.binary_cross_entropy(y, y_pred)
            self.losses.append(loss)

            error = y_pred - y

            dw = (1 / n_samples) * (X.T @ error)
            db = np.mean(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            print(f"Run #{_} - Current ")

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)