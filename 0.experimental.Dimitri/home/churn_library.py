"""
Author: Dimitri 
Date Created: 2023-02-23

This scripts provides several functions
related to the credit churn data analysis
"""


# import libraries
#import shap
#import joblib
import os
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report
import os
os.environ['QT_QPA_PLATFORM']='offscreen'



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # create a folder to store images
    if not os.path.exists('images'):
        os.makedirs('images')

    plt.figure(figsize=(20,10))
    df['Churn'].hist()
    plt.savefig('images/churn_distribution.png')
    plt.close()

    plt.figure(figsize=(20,10))
    df['Customer_Age'].hist()
    plt.savefig('images/customer_age_distribution.png')
    plt.close()

    plt.figure(figsize=(20,10))
    sns.countplot(x='Marital_Status',
                  hue='Churn', data=df)
    plt.savefig('images/marital_status_churn.png')
    plt.close()

    plt.figure(figsize=(20,10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/total_trans_ct_distribution.png')
    plt.close()

    plt.figure(figsize=(20,10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('images/heatmap.png')
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        groups = df.groupby(col).mean()[response]
        new_col_name = col + '_Churn'
        df[new_col_name] = df[col].map(groups)
    return df



def perform_feature_engineering(df, response):
    """
    Perform feature engineering on the input data.

    Args:
        df (pandas DataFrame): Input data.
        response (str): Name of the response variable.

    Returns:
        X_train (pandas DataFrame): Training data.
        X_test (pandas DataFrame): Testing data.
        y_train (pandas DataFrame): Training response data.
        y_test (pandas DataFrame): Testing response data.
    """
    # Encode categorical features
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df_encoded = encoder_helper(df, cat_columns, response)

    # Drop original categorical columns and normalize continuous features
    df_encoded.drop(cat_columns, axis=1, inplace=True)
    norm_cols = df_encoded.columns[df_encoded.columns != response]
    df_encoded[norm_cols] = normalize(df_encoded[norm_cols])

    # Split the data into training and testing sets
    X = df_encoded.drop(response, axis=1)
    y = df_encoded[response]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test



def classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    train_lr_report = classification_report(y_train, y_train_preds_lr, output_dict=True)
    train_rf_report = classification_report(y_train, y_train_preds_rf, output_dict=True)
    test_lr_report = classification_report(y_test, y_test_preds_lr, output_dict=True)
    test_rf_report = classification_report(y_test, y_test_preds_rf, output_dict=True)

    # create figure and axis objects with subplots()
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5)

    # plot each report on a subplot
    axs[0, 0].axis('off')
    axs[0, 0].table(cellText =train_lr_report.values(),
                colLabels=train_lr_report.keys(),
                rowLabels=['precision', 'recall', 'f1-score', 'support'],
                loc='center')
    axs[0, 0].set_title('Train Logistic Regression')

    axs[0, 1].axis('off')
    axs[0, 1].table(cellText =train_rf_report.values(),
                colLabels=train_rf_report.keys(),
                rowLabels=['precision', 'recall', 'f1-score', 'support'],
                loc='center')
    axs[0, 1].set_title('Train Random Forest')

    axs[1, 0].axis('off')
    axs[1, 0].table(cellText =test_lr_report.values(),
                colLabels=test_lr_report.keys(),
                rowLabels=['precision', 'recall', 'f1-score', 'support'],
                loc='center')
    axs[1, 0].set_title('Test Logistic Regression')

    axs[1, 1].axis('off')
    axs[1, 1].table(cellText =test_rf_report.values(),
                colLabels=test_rf_report.keys(),
                rowLabels=['precision', 'recall', 'f1-score', 'support'],
                loc='center')
    axs[1, 1].set_title('Test Random Forest')

    # save the figure as an image
    plt.savefig('./images/classification_report.png')


def feature_importance_plot(model, X_data, output_pth):
    """
    Creates and stores the feature importances in pth

    Parameters:
    -----------
    model: object
        Model object containing feature_importances_
    X_data: pd.DataFrame
        Pandas dataframe of X values
    output_pth: str
        Path to store the figure

    Returns:
    --------
    None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(range(X_data.shape[1]), importances[indices])
    ax.set_xticks(range(X_data.shape[1]))
    ax.set_xticklabels(names, rotation=90)
    ax.set_title("Feature Importances")
    fig.tight_layout()

    plt.savefig(output_pth)


if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    file_path = 'data/bank_data.csv'
    print("PATH "+os.path.join(current_directory, file_path))
    df = import_data(os.path.join(current_directory, file_path))
    perform_eda(df)
