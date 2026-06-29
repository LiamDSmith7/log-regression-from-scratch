
## Project Overview:
This project aimed to create a logistic regression model from scratch, in order to perform binary classification of raisin types in the raisin dataset from Kaggle. 

Given the sentence in the provided task document: 'using only built-in Python and NumPy', I have made the decision to build the repository on the assumption that only built-in Python and NumPy can be used in the model, preprocessing, training, and metrics - and therefore I avoided packages like Pandas - but I have assumed I can use packages like Jupyter, PyTest and MatPlotLib for testing the model, i.e. development tools. This project was built using Python 3.12.7.

Preprocessing was separated into its own file, so that the model's class is just for computing values. 

The model was designed to handle NumPy objects, and currently does not handle other types. Future upgrades could include pandas compatibility. 

The model was ran on the Raisin Dataset (See: data/Raisin_Dataset.csv, or: https://www.kaggle.com/datasets/nimapourmoradi/raisin-binary-classification), and the results were compared against the scikit-learn logistic regression model's performance on the same data. The results are documented below. 

## Results: 
Logistic Regression From Scratch, Baseline Model:
Accuracy: 86%;
Precision: 87%;
Recall: 86%;
F1 Score: 87%;

scikit-learn Logistic Regression:
Accuracy: 84%;
Precision: 87%;
Recall: 83%;
F1 Score: 85%


The from-scratch model performed similarly to the scikit-learn logistic regression model, with slightly stronger results on this particular train/test split. This comparison was used as a sanity check rather than as a definitive benchmark.

A second version of the model was created introducing L2 regularisation, and can be found in src/model_with_reg.py. The results are shown in the exploration.ipynb file. In short, regularisation was tested using multiple L2 regularisation strengths. The best F1 score improved from 86.7% without regularisation to 87.4% with regularisation, while accuracy remained broadly unchanged. This suggests that regularisation provided a modest improvement, but there is no strong evidence that the baseline model was heavily overfitting. Further validation, such as cross-validation, would be needed to confirm this.


## Installation

Create a virtual environment and, from the project root, install the required packages:

```bash
python3 -m venv log-reg-venv
source log-reg-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The project expects the raisin dataset to be stored at:

```text
data/Raisin_Dataset.csv
```

## Running the Project

This project was developed using Python 3.12.7. Other Python 3 versions may work, but Python 3.12.7 was the version used when creating and testing this repository.

The recommended way to run the project is from the terminal using `main.py`.

From the project root, run:

```bash
python main.py
```

This will load the data, preprocess it, train the logistic regression model, generate predictions, and print the evaluation metrics.

The notebook in `notebooks/exploration.ipynb` can also be used to inspect the workflow step by step, compare model versions, and view exploratory results. Someone may do the same in their own, running step by step. However, `main.py` is the main reproducible entry point for running the project.


## Running Tests

To run the unit tests:

```bash
pytest
```
This works due to the prescence of pytest.ini, which sets the project root for pytest.


## Example Workflow

* Load the CSV with NumPy using `load_csv_with_headers`.
* Split the target variable from the feature variables using `split_features_and_target`.
* Encode the labels as `0` and `1` using `encode_labels`.
* Split the dataset into training and test sets using `train_test_split`.
* Scale the feature variables using `standardise_train_test`.
* Create the model, for example:

```python
model = LogisticRegressionScratch(
    learning_rate=0.01,
    n_iters=1000
)
```

* Train the model on the training data:

```python
model.fit(X_train_scaled, y_train)
```

* Generate predictions and evaluate the model using the metric functions in `metrics.py`.
* Optionally, run the L2 regularised version using `model_with_reg.py` and the `LogisticRegressionScratchWithReg` class.


## Key Design Decisions:
- Implemented logistic regression from scratch using built-in Python and NumPy only, to satisfy the provided task document.
- Kept preprocessing outside the model class to separate data preparation from model training.
- Used binary cross-entropy as the objective function because the task is binary classification and the model outputs probabilities. Binary cross-entropy penalises incorrect probability estimates and gives gradient descent a clear loss value to minimise during training.
- Used gradient descent to optimise weights and bias as I understood this concept well, and therefore using this technique would both achieve the project objective, and do so within the four hour time limit. 
- Stored losses during training to inspect whether the model is learning.
- Scaled features before training because logistic regression with gradient descent is sensitive to feature scale.
- Did not regularise the bias term because regularisation is intended to penalise large feature weights. The bias only controls the model’s baseline prediction and is not associated with any individual input feature.
- Used a fixed 0.5 probability threshold for binary classification.
- Compared results against scikit-learn only as a sanity check, not as part of the main implementation.
- Added optional L2 regularisation to test whether penalising large weights improves generalisation.

## This Specific Model's Assumptions:
The model expects processed numeric NumPy arrays as input.

The preprocessing functions in preprocessing.py handle loading the CSV, splitting features and labels, encoding the target variable, creating the train/test split, and scaling the features. The model itself does not perform data cleaning, missing-value handling, outlier detection, or automatic type conversion. These steps are expected to be handled before fitting the model.

## General Logistic Regression Assumptions

Logistic regression assumes that each row in the dataset is independent. In other words, one raisin measurement should not directly depend on another.

For continuous input features, logistic regression assumes that each feature has a roughly linear relationship with the model’s underlying prediction score.

The model also works best when input features are not too strongly correlated with each other. If two columns contain almost the same information, this can make the model less stable.

Very unusual outliers can also affect the model strongly, so extreme values should be checked before training (The current preprocessing scales features, but it does not automatically detect or remove outliers.).

Finally, the dataset should contain enough examples for the number of input features. If there are too many features and too few examples, the model may overfit and perform poorly on new data.



## Testing and Maintainability

This repository includes lightweight unit tests using `pytest`.

So far, tests have been added for:

* core metric functions, including accuracy, precision, recall, F1 score and confusion matrix
* basic model behaviour, including checking that the model returns binary predictions
* prediction probabilities, checking that outputs are between 0 and 1
* model fitting, checking that the correct number of feature weights is created

These tests are intentionally simple, but they help make the project more maintainable. If the code is updated later, the tests can be run quickly to check that core functionality has not been broken.

To run the tests, activate your virtual environment (If you haven't already), then run pytest from the project root:

```bash
pytest
```

This will run a batch of unit tests on core metric functions, including accuracy, precision, recall, F1 score and confusion matrix, and also test the logistic regression object's features. This batch of tests can be expanded at the user's discretion.

The project root should be the folder containing `src/`, `tests/`, `main.py`, and `pytest.ini`.

If all tests pass, pytest will show a success message. If a test fails, pytest will show which function or behaviour needs attention.

## Model/Repo Limitations: 
- No cross-validation within this version of the repository. 
- Basic gradient descent optimiser was used to meet the project's objectives on time.
- Here a basic stop criteria was used, simply to stop after N number of iterations.
- No automated hyperparameter search
- Results are based on a single train/test split, so scores may vary with a different split.
- scikit-learn comparison is approximate because it uses a different optimiser.

## Future Upgrades

The provided task document mentions several areas that would be valuable to explore beyond the four-hour time limit. Given more time, I would extend the project by adding:

* **Improved stopping criteria**, such as stopping training when the loss stops improving, rather than always training for a fixed number of iterations.
* **Cross-validation** to evaluate the model more robustly than a single train/test split.
* **Model saving and loading**, so trained weights, bias, scaling mean, and scaling standard deviation can be reused for future predictions.
* **A small prediction API**, for example using a simple Python web server, so new raisin measurements could be submitted and classified.
* **Basic model monitoring**, including tracking prediction accuracy over time if new labelled data becomes available.
* **Simple data drift checks**, such as comparing new feature distributions against the original training data.
* **Experiment tracking**, for example recording learning rate, iterations, regularisation strength, and evaluation metrics across runs.
* **A small containerised version of the project**, so the model can be run consistently in different environments.
* **Improved explainability**, such as reporting feature weights to show which measurements most influence the prediction.
* **Alternative optimisation methods**, such as Newton’s method or stochastic gradient descent, to compare convergence speed and model performance against the current batch gradient descent approach.
