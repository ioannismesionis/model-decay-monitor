# Import packages
import pytest

import os
import sys
import pandas as pd
import numpy as np

# Append path for CI/CD to find imports due to remote machines
os.chdir(os.getcwd())
sys.path.append(os.getcwd())

# Import packages
from eztools.operations import ConfigReader
from src.etl.get_data import read_csv_data

@pytest.fixture
def non_csv_path():
    # Capture the path to store the data
    csv_path = os.path.join(os.getcwd(), 'src/data/assets/df_dummy_raw.xlsx')

    pd.DataFrame({'salary': [50000, np.nan, np.nan],
                  'work_hours': [8, 7, 4],
                  'name': ['Peterson', 'Andersen', 'David']}).to_csv(csv_path, index = False)

    return csv_path


# 1. Create a dummy dataframe to be stored as a csv with no na values
# Capture the path to store the data
csv_clean_path = os.path.join(os.getcwd(), 'src/data/assets/df_dummy_clean.csv')
pd.DataFrame({'age': [50, 20, 40, 70],
              'name': ['Peterson', 'Andersen', 'David', 'Gidd']}).to_csv(csv_clean_path, index = False)

# 2. Create a dummy dataframe to be stored as a csv with no null values
# Capture the path to store the data
csv_raw_path = os.path.join(os.getcwd(), 'src/data/assets/df_dummy_raw.csv')
pd.DataFrame({'salary': [50000, np.nan, np.nan],
              'work_hours': [8, 7, 4],
              'name': ['Peterson', 'Andersen', 'David']}).to_csv(csv_raw_path, index = False)



class TestReadCSVData(object):
    def test_csv_format(self, non_csv_path):
        """ Test to ensure file format is checked """
        # Try to read from the file path
        with pytest.raises(ValueError) as expecption_info:
            _ = read_csv_data(non_csv_path)

        # Check if ValueError contains correct message
        message = 'File extension error: Expected: .csv - Received: xlsx'
        
        assert expecption_info.match(message)
    
    
    @pytest.mark.parametrize('file_path', [(csv_clean_path), (csv_raw_path)])
    def test_pd_dataframe_format(self, file_path):
        """ Test if output is a dataframe """
        # Read from the file path
        df = read_csv_data(file_path)
        
        assert isinstance(df, pd.DataFrame), 'Expected Instance: {0}, Actual: {1}'.format(pd.DataFrame, type(df))
    
    
    @pytest.mark.parametrize('file_path, expected_cols', [(csv_clean_path, ['age', 'name']), (csv_raw_path, ['salary', 'work_hours', 'name'])])
    def test_df_columns(self, file_path, expected_cols):
        """ Test for the correct columns """
        # Read from the file path
        df = read_csv_data(file_path)
        
        # Get actual columns
        actual_cols = list(df.columns)
        
        assert actual_cols == expected_cols, ' Columns Expected: {0}, Actual: {1}'.format(expected_cols, actual_cols)
        
    
    @pytest.mark.parametrize('file_path, expected_shape', [(csv_clean_path, (4, 2)), (csv_raw_path, (3, 3))])
    def test_df_shape(self, file_path, expected_shape):
        """ Test for correct data shape """
        # Read from the file path
        df = read_csv_data(file_path)
        
        actual_shape = df.shape
        
        assert actual_shape == expected_shape, 'Shape Expected: {0}, Actual: {1}'.format(expected_shape, actual_shape)
