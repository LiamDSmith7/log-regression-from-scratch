import numpy as np
import pytest

from src.model import LogisticRegressionScratch


def test_model_outputs_binary_predictions():
    X = np.array([
        [0],
        [1],
        [2],
        [3],
    ])

    y = np.array([0, 0, 1, 1])

    model = LogisticRegressionScratch(
        learning_rate=0.1,
        n_iters=1000
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert set(predictions).issubset({0, 1})


def test_predict_proba_outputs_probabilities_between_0_and_1():
    X = np.array([
        [0],
        [1],
        [2],
        [3],
    ])

    y = np.array([0, 0, 1, 1])

    model = LogisticRegressionScratch(
        learning_rate=0.1,
        n_iters=1000
    )

    model.fit(X, y)
    probabilities = model.predict_proba(X)

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_model_creates_correct_number_of_weights():
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ])

    y = np.array([0, 0, 1, 1])

    model = LogisticRegressionScratch()
    model.fit(X, y)

    assert len(model.weights) == X.shape[1]