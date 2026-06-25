
Given the sentence in the requirements document: 'using only built-in Python and NumPy', I have built the repository on the assumption that only built-in Python and Numpy can be used in the model, preprocessing, training, and metrics - and therefore I avoided packages like Pandas - but I can use packages like Jupyter, PyTest and MatPlotLib for testing the model, i.e. development tools. 

I'll seperate preprocessing out into its own file, so that the model's class is just for computing values. 

Model Assumptions:
The model is assuming processed data, i.e. it is presuming the X_train and X_test variables are appropriately scaled. Functions are provided in preprocessing.py for this, however the model does not resolve issues with this, and the user is expected to do the appropriate cleaning before running the analysis. 

