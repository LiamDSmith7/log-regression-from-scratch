"""
Entry point for training and evaluating the logistic regression model.

Run from the terminal with:
    python train.py
"""

from pathlib import Path

from src.preprocessing import (
    load_csv_with_headers,
    split_features_and_target,
    encode_labels,
    train_test_split,
    standardise_train_test,
)

from src.model import LogisticRegressionScratch

from src.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix_labelled,
)


def main():
    data_path = Path("data") / "Raisin_Dataset.csv"

    headers, data = load_csv_with_headers(data_path)
    X, y_raw = split_features_and_target(data)
    y, label_names = encode_labels(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_seed=42
    )

    X_train_scaled, X_test_scaled, mean, std = standardise_train_test(
        X_train, X_test
    )

    model = LogisticRegressionScratch(
        learning_rate=0.01,
        n_iters=1000,
    )

    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)

    print("Labels:", label_names)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
    print()
    print(confusion_matrix_labelled(y_test, preds))


if __name__ == "__main__":
    main()