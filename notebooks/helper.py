import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE



def test():
    print('ok')

def plot_vars(df):
    # visualize the target variable
    fig, ax = plt.subplots()
    ax.hist(df['Attrition_Flag'])
    ax.set_xlabel('Attrition Flag')
    ax.set_ylabel('Count')
    plt.show()

def plot_num_features(df):
    # visualize the numerical features
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(num_cols):
            ax.hist(df[num_cols[i]])
            ax.set_xlabel(num_cols[i])
    plt.show()

def plot_cat_features(df):
    # visualize the categorical features
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(cat_cols):
            ax.hist(df[cat_cols[i]])
            ax.set_xlabel(cat_cols[i])
    plt.show()


def prepro_separate_target(df):
    # separate features and target
    X = df.drop('Attrition_Flag', axis=1)
    y = df['Attrition_Flag']
    return X, y

def prepro_transform(X):
    # define numeric and categorical column transformers
    num_cols = X.select_dtypes(include=['int', 'float']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(drop='first')
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols),('cat', cat_transformer, cat_cols)])

    X_preprocessed=preprocessor.fit_transform(X)
    feature_names = num_cols.tolist() + preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)
    return X_preprocessed


class DataProcessor:
    def __init__(self, csv_file_path='../data/customer_churn_data.csv'):
        self.df = pd.read_csv(csv_file_path,index_col=0)
        self.isTransformed=False

        self.scaler = StandardScaler()
        self.encoder=OneHotEncoder(drop='first')

        self.X = self.df.drop('Attrition_Flag', axis=1)
        self.y = self.df['Attrition_Flag']
        self.num_cols = self.X.select_dtypes(include=['int', 'float']).columns
        self.cat_cols = self.X.select_dtypes(include=['object']).columns

        self.split_train_test()
        #self.balance()
    
    def split_train_test(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
    

    def scale(self, data, reverse=False):
        scaler = self.scaler()
        scaler.fit(data[self.transformnum_cols])
        scaled = scaler.transform(data[self.num_cols])
        scaled = pd.DataFrame(scaled, columns=self.num_cols)

    def transform(self,operation,reverse=False):

        self.preprocessor = ColumnTransformer(transformers=[('num', self.scaler, self.num_cols),('cat', self.encoder, self.cat_cols)])
        data_splits = [self.X_train, self.X_test, self.X]
        for i, data in enumerate(data_splits):
            data_splits[i] = self.preprocessor.fit_transform(data)
        feature_names = self.num_cols.tolist() + self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.cat_cols).tolist()
        self.X_train, self.X_test, self.X = [pd.DataFrame(data, columns=feature_names) for data in data_splits]
        self.isTransformed=True

    
    def reverse_transform(self):

        if self.isTransformed:
            for data in [self.X_test,self.X_train,self.X]:
                data=self.scaler.inverse_transform(data)
        self.isTransformed=False

    def balance(self):
        sm = SMOTE(random_state=42)
        self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)




