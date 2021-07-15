# Import required packages
import pandas as pd

# Sklearn
from sklearn.model_selection import train_test_split

# Import logger
import logging
logger = logging.getLogger('L&L')


def get_train_test_set(df, response, encode = True, pos_class = None, random_state = 0):
    """
    Get the train and test sets to train a ML model
    
    Args:
        df (pd.DataFrame): The data
        response (str): The name of the response column
        encode (boolean): To encode the response variable for a binary classification problem
        pos_class (str): The name of the positive class - Only applies if encode = True
        random_state (int): Random state to reproduce the same split (Similar to Sklearn)
        
    Returns:
        4x pd.DataFrame: X_train - Train dataset (without response)
                         X_test  - Test dataset (without response)
                         y_train - Response for train dataset
                         y_test  - Response for test dataset
    """
    df = df.copy()
    
    logger.info(f'Capturing the response col: {response}')
    
    # Split in X and y
    X = df.drop(response, axis = 1)
    y = df[response]
    
    # Encode for binary
    if encode:
        # Check if positive class has not been specified
        if not pos_class:
            raise ValueError('No positive class is selected')
        
        # Check if the positive class value is part of the values of the response
        if pos_class not in df[response].unique():
            raise ValueError('The positive class value {0} is not present in the values of the response'.format(pos_class))
            
        # Else, perform the encoding
        logger.info('Performing binary encoding in the response')
        
        y = y.map(lambda x: 1 if x == pos_class else 0)
    
    logger.info('Split the data into train and test set')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state)

    return X_train, X_test, y_train, y_test
