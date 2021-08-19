# Import packages
import pytest

import os
import sys
import math

# Append path for CI/CD to find imports due to remote machines
os.chdir(os.getcwd())
sys.path.append(os.getcwd())

cwd = os.getcwd()

from src.etl.get_data import read_csv_data
from src.etl.get_missing_values import get_df_na, get_na_columns, impute_nan
from src.etl.get_train_test_set import get_train_test_set

# 1. Read the sample data we are going to use for our tests
data_path = os.path.join(cwd, 'src/data/assets/df_wine.csv')
df_wine = read_csv_data(data_path)

# 2. Missing values
# 2.1 Capture columns that need imputation
df_na = get_df_na(df_wine)
COLS_TO_IMPUTE = get_na_columns(df_na)

# 2.2 Impute na values 
df_wine = impute_nan(df_wine, cols = COLS_TO_IMPUTE, replacement = 'mean')

class TestGetTrainTestSet(object):
    def test_X_test_split(self):
        """ Test the Test split sizes of the initial data frame """

        # Split the data frame to train & test
        _ , X_test, _ , y_test = get_train_test_set(df_wine, response = 'wine_colour', pos_class = 'white')
        
        # Capture X_test shape
        expected_X_test_shape = math.ceil(0.25 * df_wine.shape[0])
        actual_X_test_shape = X_test.shape[0]

        assert actual_X_test_shape == expected_X_test_shape, 'Expected X_Test rows: {0}, Received: {1}'.format(expected_X_test_shape, actual_X_test_shape)
        
        # Capture y_test shape
        expected_y_test_shape = expected_X_test_shape
        actual_y_test_shape = len(y_test)
        
        assert expected_y_test_shape == actual_y_test_shape, 'Expected y_test shape: {0}, Received: {1}'.format(expected_y_test_shape, actual_y_test_shape) 
        

    def test_X_train_split(self):
        """ Test the Train split sizes of the initial data frame """

        # Split the data frame to train & test
        X_train, _, y_train, _ = get_train_test_set(df_wine, response = 'wine_colour', pos_class = 'white')
        
        # Capture X_train shape
        expected_X_train_shape = math.floor(0.75 * df_wine.shape[0])
        actual_X_train_shape = X_train.shape[0]

        assert actual_X_train_shape == expected_X_train_shape, 'Expected X_Test rows: {0}, Received: {1}'.format(expected_X_train_shape, actual_X_train_shape)
        
        # Capture y_test shape
        expected_y_train_shape = expected_X_train_shape
        actual_y_train_shape = len(y_train)
        
        assert expected_y_train_shape == actual_y_train_shape, 'Expected y_test shape: {0}, Received: {1}'.format(expected_y_train_shape, actual_y_train_shape) 


    def test_label_encoding(self):
        """ Test the encoding of the variables """
        _, _, y_train, y_test = get_train_test_set(df_wine, response = 'wine_colour', pos_class = 'white')
        
        positive_records = df_wine[df_wine['wine_colour'] == 'white'].shape[0]
        pos_cls_records = (y_train == 1).sum() + (y_test == 1).sum() 

        negative_records = df_wine[df_wine['wine_colour'] == 'red'].shape[0]
        neg_cls_records = (y_train == 0).sum() + (y_test == 0).sum()
    
        assert positive_records == pos_cls_records, 'Positive label encoding: Expected {0}, Received {1}'.format(positive_records, pos_cls_records)
        assert negative_records == neg_cls_records, 'Negative label encoding: Expected {0}, Received {1}'.format(negative_records, neg_cls_records)


    @pytest.mark.parametrize('pos_class, expected_message', 
                            [(None, 'No positive class is selected'), 
                             ('misspelled', 'The positive class value misspelled is not present in the values of the response')])
    def test_expected_fails_for_encoding(self, pos_class, expected_message):
        """ Test the encoding of the variables """
        
        # Capture the columns to impute
        with pytest.raises(ValueError) as expecption_info:
             _, _, _, _ = get_train_test_set(df_wine, response = 'wine_colour', encode = True, pos_class = pos_class)
                
        assert expecption_info.match(expected_message)

    
        













# from src.etl.get_data import read_csv_data
# from src.etl.get_missing_values import get_df_na, get_na_columns, impute_nan
# from src.etl.get_train_test_set import get_train_test_set

# Import packages
# from eztools.operations import ConfigReader

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

# ## Tests to be done for splitting the data into train and test
# # 1. Split is done successfully by validating shape
# X_train_records = X_train.shape[0]
# X_test_records = X_test.shape[0]

# y_train_records = len(y_train)
# y_test_records = len(y_test)

# total = df.shape[0]

# @pytest.mark.parametrize("train_records, test_records, exp_records", [(X_train_records, X_test_records, total), (y_train_records, y_test_records, total)])
# def test_df_expected_records_split(train_records, test_records, exp_records):
#     """ Test if expected columns are present """
#     assert train_records + test_records == exp_records, 'There are records missing in the split'


# # 2. Test positive class encoding was done successfully 
# # Check encoding of labels
# white_records = df[df['wine_colour'] == 'white'].shape[0]
# pos_cls_records = (y_train == 1).sum() + (y_test == 1).sum() 

# negative_records = df[df['wine_colour'] == 'red'].shape[0]
# negative_cls_records = (y_train == 0).sum() + (y_test == 0).sum() 

# @pytest.mark.parametrize("nominal_records, encoded_records", [(white_records, pos_cls_records), (negative_records, negative_cls_records)])
# def test_label_encoding(nominal_records, encoded_records):
#     """ Test if label encoding was done successfully """
#     assert nominal_records == encoded_records, 'There were issues in the label encoding; rows missing'