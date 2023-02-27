import os
import pandas as pd

def get_df():
    # Get the path to the directory containing this script
    script_dir = os.getcwd()

    # Get the parent directory of the script directory
    parent_dir = os.path.dirname(script_dir)
    #print(parent_dir)

    # Define the path to the file relative to the current working directory
    file_path = os.path.join(parent_dir, 'data', 'raw_data', 'churn.csv')

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path,index_col=0)
    df=df.iloc[:,:-2]


    return df
if __name__ == "__main__":
    pass