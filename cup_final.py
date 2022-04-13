import pandas as pd
from mlp import MLPRegressor


# Read training set

file_path = 'data/ML-CUP20-TR.csv'
data = pd.read_csv(file_path, sep=',', header=None)
y = data[[11, 12]].to_numpy()
X = data[[1,2,3,4,5,6,7,8,9,10]].to_numpy()

n_attribute = X.shape[1]
n_output_unit = y.shape[1]

#################################################

# Train final model and plot the learning curve

param = {
    'learning_rate': 0.02418422806710919,
    'n_unit_per_hidden_layer': [20, 15],
    'momentum': 0.6517810453142732,
    'l2': 0.00010961276457171888,
    'hidden_activation_function': 'sigm'
}
model = MLPRegressor(n_attribute, n_output_unit, shuffle=True, max_iter=300, batch_size=10)
model.set_params(**param)
model.train(X, y)

#################################################

# Predict blind test set

file_path = 'data/ML-CUP20-TS.csv'
data = pd.read_csv(file_path, sep=',', header=None)
X_test = data[[1,2,3,4,5,6,7,8,9,10]].to_numpy()

y_pred = model.predict(X_test)
for i in range(len(y_pred)):
    print("%d,%s,%s" % (i+1, y_pred[i][0], y_pred[i][1]))
