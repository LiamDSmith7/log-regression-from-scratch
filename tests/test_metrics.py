import numpy as np

from src.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    confusion_matrix_labelled,
)


def test_accuracy_score():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    assert accuracy_score(y_true, y_pred) == 0.75


def test_precision_score():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    assert precision_score(y_true, y_pred) == 1.0


def test_recall_score():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    assert recall_score(y_true, y_pred) == 0.5


def test_f1_score():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    assert round(f1_score(y_true, y_pred), 2) == 0.67


def test_confusion_matrix():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    expected = np.array([
        [2, 0],
        [1, 1]
    ])

    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), expected)


def test_confusion_matrix_labelled():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    result = confusion_matrix_labelled(y_true, y_pred)

    assert "Confusion Matrix" in result
    assert "Predicted 0" in result
    assert "Predicted 1" in result
    assert "Actual 0" in result
    assert "Actual 1" in result