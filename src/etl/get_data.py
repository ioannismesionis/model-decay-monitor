# Import required packages
import pandas as pd

import logging
logger = logging.getLogger('L&L')


def read_csv_data(path):
    """
    Read a .csv file given a file path
    
    Args:
        path (str): full path of the data location
        
    Returns:
        pd.DataFrame: The data that was read
    """
    
    # Ensures that file extension is a .csv
    file_extension = path.split('.')[-1]
    
    if file_extension != 'csv':
        raise ValueError('File extension error: Expected: .csv - Received: {0}'.format(file_extension))
        
    logger.info(f'Reading the data from path: {path}')
    
    return pd.read_csv(path)