import numpy as np
import pandas as pd
from grid_search import GridSearch
from mlp import MLPRegressor
from util import *
import time


# Read data

file_path = 'data/ML-CUP20-TR.csv'
data = pd.read_csv(file_path, sep=',', header=None)
y = data[[11, 12]].to_numpy()
X = data[[1,2,3,4,5,6,7,8,9,10]].to_numpy()

n_attribute = X.shape[1]
n_output_unit = y.shape[1]

# Get internal test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

#################################################

model = MLPRegressor(n_attribute, n_output_unit, shuffle=True, max_iter=300, batch_size=10)

hyperparameters = {
    'learning_rate': (0.05 - 0.01) * np.random.random_sample((2,)) + 0.01, # eta in range [0.01, 0.05]
    'n_unit_per_hidden_layer': np.random.randint(20-3+1, size=(3,2)) + 3,  # 2 layers with num unit between [3, 20]
    'momentum': (0.8 - 0.5) * np.random.random_sample((2,)) + 0.5,         # alpha in range [0.5, 0.8],
    'l2': (0.0003 - 0.0001) * np.random.random_sample((2,)) + 0.0001,      # lambda in range [0.0001, 0.0003],
    'hidden_activation_function': ['sigm'],     # for hidden layer
}

grid_search = GridSearch(model, hyperparameters)
grid_search.train(model, X_train, y_train, 4, 'mee')
