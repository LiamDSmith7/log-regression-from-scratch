


Given the sentence in the requirements document: 'using only built-in Python and NumPy', I have built the repository on the assumption that only built-in Python and Numpy can be used in the model, preprocessing, training, and metrics - and therefore I avoided packages like Pandas - but I can use packages like Jupyter, PyTest and MatPlotLib for testing the model, i.e. development tools. 

I'll seperate preprocessing out into its own file, so that the model's class is just for computing values. 

Results: The logistic model from scratch performed marginally better than the SKlearn logstic regression model when measured by F1 Score and Accuracy. Seeing improvements from 84% accuracy and 85% F1 Score in the SKLearn model, and 86% accuracy and 87% F1 Score in the model here created from scratch. 

A second version of the model was created introducing L2 regularisation. The results are shown in the exploration.ipynb file. In short, regularisation, when tested at multiple regularisation strengths, produced small (from 86.7 to 87.4% in F1 Score, no change of accuracy) improvements. This suggests that the original model was not overfitting on the original training set. 

Model Assumptions:
The model is assuming processed data, i.e. it is presuming the X_train and X_test variables are appropriately scaled. Functions are provided in preprocessing.py for this, however the model does not resolve issues with this, and the user is expected to do the appropriate cleaning before running the analysis. 

