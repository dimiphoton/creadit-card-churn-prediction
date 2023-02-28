import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    # separate features and target
    X = df.drop('Attrition_Flag', axis=1)
    y = df['Attrition_Flag']
    
    # define numeric and categorical column transformers
    num_cols = X.select_dtypes(include=['int', 'float']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(drop='first')
    
     # apply column transformers
    #preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols),
    #                                               ('cat', cat_transformer, cat_cols)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #X_train = preprocessor.fit_transform(X_train)
    #feature_names = num_cols.tolist() + preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    #X_train = pd.DataFrame(X_train, columns=feature_names)
    #X_test = preprocessor.transform(X_test)
    #X_test = pd.DataFrame(X_test, columns=feature_names)
    
    # resample the data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test
