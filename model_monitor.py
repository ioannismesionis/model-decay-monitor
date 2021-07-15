# Import packages not present in prod/base/1.1 (e.g. evidently)
import os
import pandas as pd
import joblib

# Load config and logger
from eztools.operations import Logger, ConfigReader
logger = Logger('/mnt/logs/', logger_name = 'L&L').get_logger()

# os.system('sudo pip install evidently')
os.system('sudo pip install plotly.express')

from evidently.dashboard import Dashboard
from evidently.tabs import (DataDriftTab,
                            CatTargetDriftTab,
                            ClassificationPerformanceTab)

# Import the mlops functions
from src.etl.mlops import create_column_mapping, generate_model_data_drift_report


def run_model_monitor():
    """
    Run the mlops to monitor model decay
    """
    try:
        # Read config.ini
        CONFIG_PATH = '/repos/poc-model-drift/src/config/config.ini'
        config = ConfigReader(CONFIG_PATH, config_tuple = False).read_config()

        # Unpack config
        REFERENCE_DATA_PATH = config['model_monitor']['reference_data_path']
        MODEL_PATH = config['model_monitor']['model_path']

        print(os.getcwd())
        
        logger.debug('Getting the model to predict on production data')

        # As part of the predictions in production, we are expected to use the pre-trained model
        lasso_model = joblib.load(MODEL_PATH)

        # Get the production data (i.e. new data that our model is expected to classify in production)
        # Note: Data has been stored by Ioannis M. for demonstrating purposes and in a real-life case that would come as a schedule job
        df_production_day1 = pd.read_pickle('./src/data/assets/df_day1.pickle')
        
        # Call the predict pipeline
        df_production_day1['prediction'] = lasso_model.predict(df_production_day1)
        df_production_day1['prediction'] = df_production_day1['prediction'].map({1: 'white', 0: 'red'})

        logger.debug('Get the ground truth data')

        # Bring in the ground truth data and drop the response because we do not have the ground truth yet for the production data
        os.system(f"hadoop fs -copyToLocal -f /analyst/data_science/poc-data-drift/df_reference.pickle /repos/poc-model-drift/")
        df_reference = pd.read_pickle(REFERENCE_DATA_PATH)
        df_reference.drop('wine_colour', axis=1, inplace=True)
        
        # Get the column mapping
        column_mapping = create_column_mapping(df_reference, prediction = 'prediction')

        # Generate the model & prediction drift in production (e.g. day 1)
        generate_model_data_drift_report(df_ref = df_reference, df_prod = df_production_day1,
                                        column_mapping = column_mapping, response_type = 'categorical')
        
        
        # Calculate the report of the model and target/predict drift
        report = Dashboard(tabs=[DataDriftTab, CatTargetDriftTab])
        report.calculate(df_reference,      # Reference data with target and/or prediction
                        df_production_day1,     # Current data with target and/or prediction
                        column_mapping=column_mapping)
        
        report.show()

        return 'Model monitor report is generated :)'

    except Exception as e:
        logger.critical(e, exc_info = True)

        return 'Error in the model monitor report'


if __name__ == '__main__':
    res = run_model_monitor()
    print(res)
