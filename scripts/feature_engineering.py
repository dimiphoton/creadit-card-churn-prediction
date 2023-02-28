from sklearn.preprocessing import OneHotEncoder

# get the list of categorical features
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# one hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore')
X_cat = encoder.fit_transform(X[cat_cols])
