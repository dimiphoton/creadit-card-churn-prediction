import pandas as pd
import joblib

def create_toy_cleaned_data():
    # generate toy cleaned data
    cleaned_data = pd.read_csv('data/raw_data.csv').head(10) # read only the first 10 rows
    cleaned_data.to_csv('data/toy_outputs/cleaned_data.csv', index=False)

def create_toy_encoded_data():
    # generate toy encoded data
    encoded_data = pd.read_csv('data/processed_data.csv').head(10) # read only the first 10 rows
    encoded_data.to_csv('data/toy_outputs/encoded_data.csv', index=False)

def create_toy_normalized_data():
    # generate toy normalized data
    normalized_data = pd.read_csv('data/processed_data.csv').head(10) # read only the first 10 rows
    normalized_data.to_csv('data/toy_outputs/normalized_data.csv', index=False)

def create_toy_selected_features():
    # generate toy selected features
    selected_features = ['age', 'credit_score', 'credit_limit', 'total_transactions']
    pd.DataFrame(selected_features, columns=['feature_name']).to_csv('data/toy_outputs/selected_features.csv', index=False)

def create_toy_engineered_features():
    # generate toy engineered features
    engineered_features = pd.read_csv('data/processed_data.csv').head(10)[['age', 'credit_score', 'credit_limit', 'total_transactions']]
    engineered_features['interaction'] = engineered_features['age'] * engineered_features['credit_score']
    engineered_features.to_csv('data/toy_outputs/engineered_features.csv', index=False)

def create_toy_trained_model():
    # generate toy trained model
    trained_model = {'model_type': 'logistic_regression', 'model_params': {'C': 0.1}}
    joblib.dump(trained_model, 'data/toy_outputs/trained_model.pkl')

def create_toy_predictions():
    # generate toy predictions
    predictions = pd.read_csv('data/processed_data.csv').head(10)
    predictions['churn_prediction'] = [0, 1, 0, 0, 1, 1, 0, 0, 0, 1]
    predictions.to_csv('data/toy_outputs/predictions.csv', index=False)

def create_toy_accuracy_score():
    # generate toy accuracy score
    with open('data/toy_outputs/accuracy_score.txt', 'w') as f:
        f.write('0.85\n')

def create_toy_roc_auc_score():
    # generate toy roc-auc score
    with open('data/toy_outputs/roc_auc_score.txt', 'w') as f:
        f.write('0.92\n')
