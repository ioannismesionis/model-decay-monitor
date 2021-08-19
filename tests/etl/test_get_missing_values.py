# Import packages
import pytest

import os
import sys
import pandas as pd
import numpy as np

# Append path for CI/CD to find imports due to remote machines
os.chdir(os.getcwd())
sys.path.append(os.getcwd())

cwd = os.getcwd()

from src.etl.get_data import read_csv_data
from src.etl.get_missing_values import get_df_na, get_na_columns, impute_nan

# Bring the sample data in from HDFS
# Note: HDFS is where data should be kept for testing purposes
os.system(f'hadoop fs -copyToLocal /analyst/data_science/my-awesome-project/df_wine.csv {cwd}/src/data/assets/')

# Read the sample data we are going to use for our tests
data_path = os.path.join(cwd, 'src/data/assets/df_wine.csv')
df_wine = read_csv_data(data_path)

class TestGetDfNa(object):
    @pytest.mark.parametrize('col', [('number_of_nan'), ('number_of_nan_prc')])
    def test_df_na_expected_columns(self, col):
        """ Test if expected columns are present """
        
        # Transform the data
        df_na = get_df_na(df_wine)

        assert col in df_na.columns, f'Column {col} is not in df_na'


    @pytest.mark.parametrize('row_name, nans_expected', [('pH', 4), ('density', 0)])
    def test_absolute_values_for_nas(self, row_name, nans_expected):
        """ Ensure the number of nan's for different columns are the ones expected"""
        
        # Transform the data
        df_na = get_df_na(df_wine)
        
        # Capture the actual number of nans for the specified row_name (i.e. column)
        nans_actual = df_na.loc[row_name, 'number_of_nan']
        
        assert nans_expected == nans_actual, 'Number of absolute nans: Expected:{0} - Actual:{1}'.format(nans_expected, nans_actual)
    
    
    @pytest.mark.parametrize('row_name, prc_nans_expected', [('pH', 0.061567), ('fixed acidity', 0.0)])
    def test_percentages_values_for_nas(self, row_name, prc_nans_expected):
        """ Ensure the number of nan's for different columns are the ones expected"""
        
        # Transform the data
        df_na = get_df_na(df_wine)
        
        # Capture the actual number of nans for the specified row_name (i.e. column)
        nans_actual = df_na.loc[row_name, 'number_of_nan_prc']
        
        # Note: Use pytest.approx due to python not being able to compare floats efficiently
        assert nans_actual == pytest.approx(prc_nans_expected, 1e-5), 'Number of absolute nans: Expected:{0} - Actual:{1}'.format(prc_nans_expected, nans_actual)
        

    @pytest.mark.parametrize('leading_col, following_col', [('pH', 'quality'), ('pH', 'alcohol')] )   
    def test_sorting_largests_na_col(self, leading_col, following_col):
        """ Test the column with the highest na values is on top due to shorting"""
        
        # Transform the data
        df_na = get_df_na(df_wine)
        
        leading_idx = list(df_na.index).index(leading_col)
        following_idx = list(df_na.index).index(following_col)
        
        assert leading_idx < following_idx, 'Sorting failed: Leading Index:{0}, Following Index:{1}'.format(leading_idx, following_idx)
    
    
# Capture columns needed
df_na = get_df_na(df_wine)
COLS_TO_IMPUTE = get_na_columns(df_na)

## Tests to be done on cols to compute the NA values
class TestGetNaColumns(object):
    
    def test_for_correct_nan_column(self):
        """ Test to ensure failure if wrong nan_col is put as an argument """

        # Capture the columns to impute
        with pytest.raises(ValueError) as expecption_info:
            _ = get_na_columns(df_na, nan_col = 'should_fail_col')

        # Check if ValueError contains correct message
        message = 'The specified nan_col:{0} does not exist as a column in the input data frame.'.format('should_fail_col')
        
        assert expecption_info.match(message)
        

    @pytest.mark.parametrize("df_na, expected_col", [(df_na, ['pH'])])
    def test_expected_cols_to_impute(self, df_na, expected_col):
        """ Test if cols_to_impute captures the expected columns """
        # Run the function to be tested
        COLS_TO_IMPUTE = get_na_columns(df_na)

        assert COLS_TO_IMPUTE == expected_col, 'Columns to be imputed: Expected:{0} - Actual:{1}'.format(expected_col, COLS_TO_IMPUTE)


    @pytest.mark.parametrize("df_na, not_expected_col", [(df_na, 'density'), (df_na, 'quality')])
    def test_for_not_present_columns_to_impute(self, df_na, not_expected_col):
        """ Test if cols_to_impute captures the expected columns """
        # Run the function to be tested
        COLS_TO_IMPUTE = get_na_columns(df_na)

        assert not_expected_col not in COLS_TO_IMPUTE, '{0} is in {1} when it should not'.format(not_expected_col, COLS_TO_IMPUTE)
    

## Tests to be done on imputes columns
class TestImputeNan(object):

    def test_valueError_in_replacement(self):
        """ Test to see if we get a value error for wrong 'replacement' text """

        with pytest.raises(ValueError) as expecption_info:
            impute_nan(df_wine, cols = 'pH', replacement = 'some_text_that_will_throw_error')

        # Check if ValueError contains correct message
        message = 'No valid selection for the "replacement" value; Please select "mean" or "median".'
        assert expecption_info.match(message)


    def test_impute_with_mean(self):
        """ Test to ensure the mean imputation works as expected """
        # Run the function to be tested
        df_wine_imputed = impute_nan(df_wine, 'pH', replacement='mean')

        expected_mean = 3.2185184044355415
        actual_mean = df_wine_imputed['pH'].mean()

        assert actual_mean == pytest.approx(expected_mean), 'Mean value for pH: Expected:{0} - Actual{1}'.format(expected_mean, actual_mean)


    def test_impute_with_median(self):
        """ Test to ensure the mean imputation works as expected """

        df_wine_imputed = impute_nan(df_wine, 'pH', replacement='median')

        expected_median = 3.21
        actual_median = df_wine_imputed['pH'].median()

        assert actual_median == pytest.approx(expected_median), 'Median value for pH: Expected:{0} - Actual{1}'.format(expected_median, actual_median)
