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

model = MLPRegressor(n_attribute, n_output_unit, shuffle=True, max_iter=200, batch_size=10)

# Best 10 models obtained from grid search
configs = [
    {
        'learning_rate': 0.02418422806710919,
        'n_unit_per_hidden_layer': [16, 17], 
        'momentum': 0.6031372464656821,
        'l2': 0.00010961276457171888,
        'hidden_activation_function': 'sigm'
    },{
        'learning_rate': 0.03781448434358182,
        'n_unit_per_hidden_layer': [17, 19],
        'momentum': 0.5141880116946248,
        'l2': 0.00017205602198221988,
        'hidden_activation_function': 'sigm'
    },
    {
        'learning_rate': 0.02418422806710919,
        'n_unit_per_hidden_layer': [20, 15],
        'momentum': 0.6517810453142732,
        'l2': 0.00010961276457171888,
        'hidden_activation_function': 'sigm'
    },
    {
        'learning_rate': 0.02418422806710919,
        'n_unit_per_hidden_layer': [16, 17],
        'momentum': 0.6517810453142732,
        'l2': 0.00010961276457171888,
        'hidden_activation_function': 'sigm'
    },
    {
        'learning_rate': 0.02634307195798071,
        'n_unit_per_hidden_layer': [ 9, 13],
        'momentum': 0.5032358991950362,
        'l2': 0.00010537854587356557,
        'hidden_activation_function': 'sigm'
    },
    {
        'learning_rate': 0.017829250667422464,
        'n_unit_per_hidden_layer': [ 9, 13],
        'momentum': 0.5032358991950362,
        'l2': 0.00010537854587356557,
        'hidden_activation_function': 'sigm'
    },
    {
        'learning_rate': 0.04329750372240989,
        'n_unit_per_hidden_layer': [10, 17],
        'momentum': 0.7333524878424191,
        'l2': 0.0001454565718568029,
        'hidden_activation_function': 'sigm'
    },
    {
        'learning_rate': 0.023877173264894774,
        'n_unit_per_hidden_layer': [15, 15],
        'momentum': 0.5099170683460086,
        'l2': 0.000125868171121606,
        'hidden_activation_function': 'sigm'
    },
    {
        'learning_rate': 0.03781448434358179,
        'n_unit_per_hidden_layer': [14,  9],
        'momentum': 0.7133375279094154,
        'l2': 0.00011999724954306384,
        'hidden_activation_function': 'sigm'
    },
    {
        'learning_rate': 0.03781448434358183,
        'n_unit_per_hidden_layer': [17, 19],
        'momentum': 0.7133375279094154,
        'l2': 0.00011999724954306384,
        'hidden_activation_function': 'sigm'
    }
]

counter = 1
for params in configs:
    print('param', params)
    model.set_params(**params)
    tr_scores, ts_scores = model.train(X_train, y_train, X_test, y_test, 'mee')
    print('MEE (TS): ', evaluate_score('mee', y_test, model.predict(X_test)))
    filename = "config%s.png" % counter
    counter += 1
    plot_learning_curve('MEE', (tr_scores, 'Training Set'), (ts_scores, 'Test Set'), filename=filename)
