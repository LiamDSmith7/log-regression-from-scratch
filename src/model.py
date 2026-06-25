import csv
import math
import json
import pickle
from pathlib import Path
import numpy as np 

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []
    
    