import numpy as np


import numpy as np

def load_csv_with_headers(path):
    '''
    Loads CSV files, and loads column headers and data seperately, so outputs two variables: the headers, then data.
    Requires only one argument: the CSV's path. 

    '''
    headers = np.genfromtxt(
        path,
        delimiter=",",
        max_rows=1,
        dtype=str
    )

    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        dtype=str
    )

    return headers, data

def split_features_and_target(data):
    '''
    Input the data (i.e. everything except headers), last column will be the target variable, all other columns are feature variables.
    Converts all X data into floats, so is specific to the raisin dataset. Edit accordingly. 
    '''
    
    X = data[:, :-1].astype(float)  # numeric feature columns
    y = data[:, -1]                 # string label column
    return X, y

def encode_labels(y):
    """
    Converts classes into 0 and 1.
    Example: Kecimen -> 0, Besni -> 1
    Produces two variables: y_encoded, the y-value array of values for testing, and unique_labels, a numpy array of the unique values
    """
    unique_labels = np.unique(y)

    if len(unique_labels) != 2:
        raise ValueError("Expected exactly two classes")

    y_encoded = np.where(y == unique_labels[0], 0, 1)

    return y_encoded, unique_labels


def train_test_split(X, y, test_size=0.2, random_seed=42):
    '''
    Produces four variables: x_train, x_test, Y_train, Y_test.
    test_size can be asdjusted by providing a float value for changing the train/test size split
    '''
    np.random.seed(random_seed)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_count = int(n_samples * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def standardise_train_test(X_train, X_test):
    '''
    Takes X_train and X_test numpy arrays and scales columns to improve linear regression performance.
    
    '''
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std == 0] = 1  # avoid divide-by-zero

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, mean, std