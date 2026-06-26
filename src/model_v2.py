import numpy as np


class LogisticRegressionScratchWithReg:
    def __init__(self, learning_rate=0.01, n_iters=1000, reg_strength=0.0):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.reg_strength = reg_strength  # lambda value for L2 regularisation
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        base_loss = -np.mean(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        )

        l2_penalty = self.reg_strength * np.sum(self.weights ** 2) / 2

        return base_loss + l2_penalty

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            loss = self.binary_cross_entropy(y, y_pred)
            self.losses.append(loss)

            error = y_pred - y

            dw = (1 / n_samples) * (X.T @ error)

            # Add L2 regularisation gradient
            dw += self.reg_strength * self.weights

            db = np.mean(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)