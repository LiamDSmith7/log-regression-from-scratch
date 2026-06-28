
## Project Overview:
This project aimed to create a logistic regression model from scratch, in order to perform binary classification of raisin types in the raisin dataset from Kaggle. 

Given the sentence in the requirements document: 'using only built-in Python and NumPy', I have made the decision to build the repository on the assumption that only built-in Python and NumPy can be used in the model, preprocessing, training, and metrics - and therefore I avoided packages like Pandas - but I have assumed I can use packages like Jupyter, PyTest and MatPlotLib for testing the model, i.e. development tools. 

Preprocessing was separated into its own file, so that the model's class is just for computing values. 

The model was designed to handle NumPy objects, and currently does not handle other types. Future upgrades could include pandas compatibility. 

The model was compared against the scikit-learn logistic regression model and the results are documented below. 

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

A second version of the model was created introducing L2 regularisation. The results are shown in the exploration.ipynb file. In short, regularisation was tested using multiple L2 regularisation strengths. The best F1 score improved from 86.7% without regularisation to 87.4% with regularisation, while accuracy remained broadly unchanged. This suggests that regularisation provided a modest improvement, but there is no strong evidence that the baseline model was heavily overfitting. Further validation, such as cross-validation, would be needed to confirm this.


## Installation
pip install -r requirements.txt

## Running the project
python main.py

## Running tests
pytest

## Example workflow:
- load CSV with NumPy, using 'load_csv_with_headers'
- Split target variable (your final column) from your feature variables using 'split_features_and_target'
- encoded labels as 0/1 using 'encode_labels'
- train/test split dataset using 
- scale your feature variables using 'standardise_train_test' for better performance
- load model e.g. 'model = LogisticRegressionScratch(
    learning_rate=0.01,
    n_iters=1000
)'
- train model on your training set e.g. 'model.fit(X_train_scaled, Y_train)'
- optional: Run again using L2 regularisation by doing the same on previous two steps, except using model_v2.py and class name is 'LogisticRegressionScratchWithReg'

## Key Design Decisions:
- Implemented logistic regression from scratch using NumPy only.
- Kept preprocessing outside the model class to separate data preparation from model training.
- Used binary cross-entropy as the objective function.
- Used gradient descent to optimise weights and bias.
- Stored losses during training to inspect whether the model is learning.
- Scaled features before training because logistic regression with gradient descent is sensitive to feature scale.
- Added optional L2 regularisation to test whether penalising large weights improves generalisation.
- Did not regularise the bias term, only the feature weights.
- Used a fixed 0.5 probability threshold for binary classification.
- Compared results against scikit-learn only as a sanity check, not as part of the main implementation.

## Model Assumptions:
The model is assuming processed data, i.e. it is presuming the X_train and X_test variables are appropriately scaled. Functions are provided in preprocessing.py for this, however the model does not resolve issues with this, and the user is expected to do the appropriate cleaning before running the analysis. 

## Testing and Maintainability

This repository includes lightweight unit tests using `pytest`.

So far, tests have been added for:

* core metric functions, such as accuracy and precision
* basic model behaviour, such as checking that the model returns binary predictions

These tests are intentionally simple, but they help make the project more maintainable. If the code is updated later, the tests can be run quickly to check that core functionality has not been broken.

To run the tests, first create a virtual environment, and activate it:

```bash
source log-reg-venv/bin/activate
```

Then run pytest from the project root:

```bash
pytest
```

The project root should be the folder containing `src/`, `tests/`, `main.py`, and `pytest.ini`.

If all tests pass, pytest will show a success message. If a test fails, pytest will show which function or behaviour needs attention.

## Model/Repo Limitations: 
- No cross-validation within this version of the repository. 
- Basic gradient descent optimiser
- No automated hyperparameter search
- Results are based on a single train/test split, so scores may vary with a different split.
- scikit-learn comparison is approximate because it uses a different optimiser.

## Future Upgrades:
- Cross-Validation features added in preprocessing.py
- Option to input pandas dataframes into the model

