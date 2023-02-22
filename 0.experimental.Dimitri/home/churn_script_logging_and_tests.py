import os
import logging
import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test data import
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as e:
        logging.error(f"Testing perform_eda: {e}")
        raise e

def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        categories = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
        response = 'Churn'
        cls.encoder_helper(df, categories, response)
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as e:
        logging.error(f"Testing encoder_helper: {e}")
        raise e

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, 'Churn')
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: Failed")
        raise err

def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, 'Churn')
        cls.train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as e:
        logging.error(f"Testing train_models: {e}")
        raise e

if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
