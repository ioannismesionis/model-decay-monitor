# Import required packages
import pandas as pd

# Import visualisation packages
from matplotlib import pyplot as plt
import seaborn as sns

import logging
logger = logging.getLogger('L&L')


def get_df_na(df):
    """
    Calculate the df that hold the missing values of all the columns in
    absolute values and by percentage.
    e.g. 4 na values in total &  68% na values.

    Every row in the newly generated dataframe corresponds to a column from the raw data
    and is sorted in descending order.
    e.g. First row contains the highest number of na values
    
    Args:
        df (pd.DataDrame): The raw data to calculate the na values from
    
    Returns:
        pd.DataFrame: Pandas dataframe with the respective {number_of_nan', 'number_of_nan_prc} columns 
    """
    df = df.copy()
    
    logger.info('Calculating the nan values for the data')
    
    # Calculate the number of na values
    df_na_count = df.isnull().sum()

    # Calculate the number of na values expressed in percentage
    df_na_prc = (df.isnull().sum() / df.shape[0]) * 100

    # Concatenate the two in a pd.DataFrame and sort the values in descending order
    df_na = pd.concat([df_na_count, df_na_prc], axis = 1).rename({0:'number_of_nan', 1: 'number_of_nan_prc'}, axis = 1)
    df_na.sort_values(by = 'number_of_nan', ascending = False, inplace = True)
    
    return df_na


def get_na_columns(df, nan_col = 'number_of_nan'):
    """
    Get the columns that have nan values.
    
    Args:
        df (pd.DataDrame): Data that has the columns of the original data as index
                           and a column that stores the counts of nan values
        nan_col (string): Name of column that contains the count of nan
    
    Returns:
        list: With the columns we need to impute the nan values
    """
    if nan_col not in df.columns:
        # Nan_col not in our columns
        raise ValueError(f'The specified nan_col:{nan_col} does not exist as a column in the input data frame.')
        
    logger.info('Getting the nan columns that have to be imputed')
    
    return [idx_col for idx_col in df.index if df.loc[idx_col, nan_col] != 0]
    
    
def plot_kdensity(df, col):
    """
    Plot kernel density 
    
    Args:
        df (pd.DataDrame): The data
        col (string): Column to be used for the plot
        
    Returns:
        None: plots the kdensity and shows the description of the column
    """
    # Ensure the column name exists
    if col not in df.columns:
        logger.debug('Wrong column name')
    else:
        plt.figure(figsize = (15, 7))

        sns.distplot(df[col].dropna(), 
                     hist = True,
                     kde = True,
                     hist_kws = {'edgecolor':'black'})
        plt.show()
        plt.close()
        
        
def impute_nan(df, cols, replacement = 'mean'):
    """
    Imputes nan values with according to what specified by the user
    
    Args:
        df (pd.DataFrame): The raw data
        cols (list): The list of variables/columns we want to impute the na values
        replacement (string): The imputation technique to be used (e.g. "mean" - default behaviour) 
    
    Returns:
        pd.DataFrame: Pandas dataframe with the imputed columns
    """
    df = df.copy()
    
    # Making the cols into a list if not a list already
    if not isinstance(cols, list):
        cols = [cols]
    
    # Iterating on the cols specified by the user
    for col in cols:
        if replacement == 'mean':
            logger.info(f'Impute column: {col}, Values filled with the {replacement}: {df[col].mean()}')
            
            # Fill with mean of the col
            df[col] = df[col].fillna(df[col].mean())
        elif replacement == 'median':
            logger.info(f'Impute column: {col}, Values filled with the {replacement}: {df[col].median()}')
            
            # Fill with mean of the col
            df[col] = df[col].fillna(df[col].median())
        else:
            # Raise a ValueError for a wrong imputation technique
            raise ValueError('No valid selection for the "replacement" value; Please select "mean" or "median".')
    
    logger.info('Imputation completed')
    
    return df
        