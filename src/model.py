import csv
import math
import json
import pickle
from pathlib import Path
import numpy as np 

class LogisticRegressionScratch:
    """
    Logistic regression classifier implemented from scratch using NumPy.

    This model performs binary classification using:
    - sigmoid activation
    - binary cross-entropy loss
    - gradient descent optimisation
    - optional L2 regularisation
    """
    
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Initialise model hyperparameters.

        Args:
            learning_rate (float): Step size used during gradient descent.
            n_iters (int): Number of training iterations.
            reg_strength (float): Strength of L2 regularisation. Use 0.0 for no regularisation.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        """
        Convert linear model scores into probabilities between 0 and 1.
        """
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y, y_pred):
        """
        Calculate binary cross-entropy loss with optional L2 regularisation.

        Args:
            y (np.ndarray): True binary labels.
            y_pred (np.ndarray): Predicted probabilities.

        Returns:
            float: Average loss value.
        """
        epsilon = 1e-15 # Using small value to prevent log(0) when calculating loss, by keeping probabilities just above 0 and below 1
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        )
        return loss

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Args:
            X (np.ndarray): Training features with shape (n_samples, n_features).
            y (np.ndarray): Binary target labels with shape (n_samples,).
        """
        
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

    def predict_proba(self, X):
        """
        Predict class probabilities for input features.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Probability of class 1 for each row.
        """
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        """
        Predict binary class labels using a 0.5 probability threshold.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted labels, either 0 or 1.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)