import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    # separate the features and target variable
    X = df.drop('Attrition_Flag', axis=1)
    y = df['Attrition_Flag']

    # get the list of numerical features
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # normalize skewed features with PowerTransformer
    pt = PowerTransformer()
    X[num_cols] = pt.fit_transform(X[num_cols])

    # scale features with StandardScaler
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # one hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_cat = encoder.fit_transform(X.select_dtypes(include=['object']))

    # concatenate numerical and categorical features
    X = pd.concat([pd.DataFrame(X[num_cols]), pd.DataFrame(X_cat.toarray())], axis=1)

    # upsample the data with SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled
