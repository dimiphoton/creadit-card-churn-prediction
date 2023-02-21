import json
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib

# Set the hyperparameters for the models
linear_params = {'normalize': True}
tree_params = {'max_depth': 5, 'min_samples_leaf': 2}

# Create a dictionary to store the hyperparameters
linear_config = {'model_params': linear_params}
tree_config = {'model_params': tree_params}

# Train and save the linear model
linear_model = LinearRegression(**linear_params)
joblib.dump(linear_model, 'linear_model.pkl')
with open('linear_config.json', 'w') as f:
    json.dump(linear_config, f)

# Train and save the tree model
tree_model = DecisionTreeRegressor(**tree_params)
joblib.dump(tree_model, 'tree_model.pkl')
with open('tree_config.json', 'w') as f:
    json.dump(tree_config, f)

# Load the linear model and configuration
with open('linear_config.json', 'r') as f:
    linear_config = json.load(f)
linear_model = joblib.load('linear_model.pkl')
linear_model.set_params(**linear_config['model_params'])

# Load the tree model and configuration
with open('tree_config.json', 'r') as f:
    tree_config = json.load(f)
tree_model = joblib.load('tree_model.pkl')
tree_model.set_params(**tree_config['model_params'])

# Use the linear model to make predictions on new data
linear_X = [[1, 2, 3], [4, 5, 6]]
linear_y = linear_model.predict(linear_X)
print(linear_y)

# Use the tree model to make predictions on new data
tree_X = [[1, 2, 3], [4, 5, 6]]
tree_y = tree_model.predict(tree_X)
print(tree_y)
