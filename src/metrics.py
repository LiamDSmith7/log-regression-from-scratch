import numpy as np

def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(y_true == y_pred)

def confusion_matrix_labelled(y_true, y_pred):
    '''
    Returns a confusion matrix string which is labelled for easier readability. 
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return (
        "Confusion Matrix:\n"
        "                 Predicted 0    Predicted 1\n"
        f"Actual 0         {tn:<12} {fp:<12}\n"
        f"Actual 1         {fn:<12} {tp:<12}"
    )

def confusion_matrix(y_true, y_pred):
    '''
    Returns a confusion matrix which is a numpy array. 
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([
        [true_negative, false_positive],
        [false_negative, true_positive]
    ])

def precision_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))

    if true_positive + false_positive == 0:
        return 0

    return true_positive / (true_positive + false_positive)

def recall_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    if true_positive + false_negative == 0:
        return 0

    return true_positive / (true_positive + false_negative)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision + recall == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)