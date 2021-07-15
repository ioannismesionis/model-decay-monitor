# Load config and logger
from eztools.operations import Logger, ConfigReader
logger = Logger('/mnt/logs/', logger_name = 'L&L').get_logger()

# Import packages
from src.etl.get_data import read_csv_data
from src.etl.get_missing_values import get_df_na, get_na_columns, impute_nan, plot_kdensity
from src.etl.get_train_test_set import get_train_test_set

# Modelling packages
from src.ml.get_lasso_model_predictions import get_lasso_model_predictions
from src.ml.get_model_accuracy import get_model_accuracy


def run_demo():
    '''
    Running a series of functions as part of the Launch & Learn session and TDD best practices
    
    Args:
        None
        
    Returns:
        dict: Dictionary with the updated "message", "results" and "classification_metrics" field that shows whether the code succeeded
    '''
    try:
        # Read config.ini
        CONFIG_PATH = '/repos/poc-model-drift/src/config/config.ini'
        config = ConfigReader(CONFIG_PATH, config_tuple = False).read_config()

        # Unpack config
        DATA_PATH = config['data']['data_path']
        RESPONSE = config['data']['response']
        POSITIVE_CLASS = config['data']['positive_class']
        
        # ML pipeline
        # Read data
        df = read_csv_data(DATA_PATH)
        
        # Missing values
        df_na = get_df_na(df)
        
        # Capture columns with nan values
        COLS_TO_IMPUTE = get_na_columns(df_na)
        
        # Impute nan values
        for col in COLS_TO_IMPUTE:
            df = impute_nan(df, cols = col, replacement = 'mean')
        
        # Modelling 
        # Split data into train and test set
        X_train, X_test, y_train, y_test = get_train_test_set(df, response = RESPONSE, pos_class = POSITIVE_CLASS)
        
        # Train lasso model & make the predictions
        y_pred = get_lasso_model_predictions(X_train, X_test, y_train)
        
        # Model evaluation
        accuracy = get_model_accuracy(y_test, y_pred)

        # Update the config message if it run successfully
        config['message'] = 'Run code successfully'
        config['result'] = 0
        config['model_accuracy'] = accuracy
    except Exception as e:
        logger.critical(e, exc_info = True)
        config['message'] = e
        config['result'] = -1
        config['model_accuracy'] = {}

    return {k: config[k] for k in ('message', 'result', 'model_accuracy')}
        


if __name__ == '__main__':
    res = run_demo()
    print(res['message'], res['result'])
    print(res['model_accuracy'])
    