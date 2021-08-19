# Import packages
import pytest

import os
import sys
import pandas as pd
import numpy as np


# Append path for CI/CD to find imports due to remote machines
os.chdir(os.getcwd())
sys.path.append(os.getcwd())

# Modelling packages
from src.etl.get_train_test_set import get_train_test_set
from src.ml.get_lasso_model_predictions import get_lasso_model_predictions

# Import classification data
from sklearn.datasets import make_classification

@pytest.fixture
def signal_classification_data():
    # Make classification data
    features, label = make_classification(n_samples = 10,     # 10 rows
                                          n_features = 3,     # 3 features
                                          n_informative = 3,  # 3 features that actually predict the output's classes
                                          n_redundant = 0,    # 0 feature that are random and unrelated to the output's classes
                                          n_classes = 2,      # Binary classification
                                          flip_y = 0,         # Do not add noise
                                          random_state = 0)

    # Create data
    y = pd.DataFrame(label).rename({0:'label'}, axis = 1)
    X = pd.DataFrame(features)
    
    # Unify the data
    df = pd.concat([X, y], axis = 1)

    return df


@pytest.fixture
def noisy_classification_data():
    # Make classification data
    features, label = make_classification(n_samples = 10,     # 10 rows
                                          n_features = 4,     # 3 features
                                          n_informative = 2,  # 3 features that actually predict the output's classes
                                          n_redundant = 2,    # 0 feature that are random and unrelated to the output's classes
                                          n_classes = 2,      # Binary classification
                                          flip_y = 0.5,       # Maximum noise
                                          random_state = 0)

    # Create data
    y = pd.DataFrame(label).rename({0:'label'}, axis = 1)
    X = pd.DataFrame(features)
    
    # Unify the data
    df = pd.concat([X, y], axis = 1)

    return df


class TestGetLassoModelPredictions(object):
    def test_on_signal_classification_data(self, signal_classification_data):
        """ Test using classification data made from sklearn """
        
        # Split the dataset
        X_train, X_test, y_train, y_test = get_train_test_set(signal_classification_data, response = 'label', encode = False)

        # Train the model and get the predictions
        y_pred_actual = get_lasso_model_predictions(X_train, X_test, y_train)
        y_pred_expected = np.array(y_test)
        
        assert all(y_pred_expected == y_pred_actual), 'Prediction expected: {0}, Actual: {1}'.format(y_pred_expected, y_pred_actual)
    
    
    def test_on_noisy_classification_data(self, noisy_classification_data):
            """ Test using classification data made from sklearn """
            
            # Split the dataset
            X_train, X_test, y_train, y_test = get_train_test_set(noisy_classification_data, response = 'label', encode = False)

            # Train the model and get the predictions
            y_pred_actual = get_lasso_model_predictions(X_train, X_test, y_train)
            y_pred_expected = np.array(y_test)
            
            assert all(y_pred_expected != y_pred_actual), 'Prediction expected: {0}, Actual: {1}'.format(y_pred_expected, y_pred_actual)










































# # Import packages
# from eztools.operations import ConfigReader

# from src.etl.get_data import read_csv_data
# from src.etl.get_missing_values import get_df_na, get_na_columns, impute_nan
# from src.etl.get_train_test_set import get_train_test_set

# # Modelling packages
# from src.ml.get_lasso_model_predictions import get_lasso_model_predictions

# # Read config.ini
# CONFIG_PATH = os.path.join(os.getcwd(), 'src/config/config.ini')
# config = ConfigReader(CONFIG_PATH, config_tuple = False).read_config()

# # Config unpack
# RESPONSE = config['data']['response']
# POSITIVE_CLASS = config['data']['positive_class']

# # Data path
# data_name = 'df_wine.csv'
# TEST_DATA_PATH = os.path.join(os.getcwd(), data_name)

# ## Run the pipeline
# # Read the data
# df = read_csv_data(TEST_DATA_PATH)

# # Get the NA values
# df_na = get_df_na(df)

# # Capture columns needed
# COLS_TO_IMPUTE = get_na_columns(df_na)

# # Impute missing values  
# df = impute_nan(df, cols = COLS_TO_IMPUTE, replacement = 'mean')

# # Split the train and test data
# X_train, X_test, y_train, y_test = get_train_test_set(df, response = RESPONSE, pos_class = POSITIVE_CLASS)

# # Train lasso model & make the predictions
# y_pred = get_lasso_model_predictions(X_train, X_test, y_train)

# ## Tests to be done for splitting the data into train and test
# # 1. Valid output
# @pytest.mark.parametrize("y_pred, min_, max_", [(y_pred, 0, 1)])
# def test_valid_output(y_pred, min_, max_):
#     """ Test if predictions are expected """
#     assert all(min_ <= y_pred), 'There are negative probabilities'
#     assert all(y_pred <= max_), 'There are probabilities above 1'
