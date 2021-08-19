# Import packages needed
import os
import pandas as pd
from datetime import datetime
import joblib

# Load config and logger
# Configure logger
logging.basicConfig(level = logging.INFO, format = '%(asctime)s :: %(levelname)s :: %(message)s')
logger = logging.getLogger(__name__)

# Import mlops packages
from src.ml.model_monitor import ModelMonitorReports, MonitorReportReader



def run_model_decay():
    """
    Run the mlops to monitor model decay
    """
    try:
        # 1. Settings
        # Read config.toml
        CONFIG_PATH = './src/config/config.toml'
        config = toml.load(CONFIG_PATH)

        # Unpack the config file
        LATEST_DRIFT_REPORT = config["json_reports_path"]["latest_drift_report_json"]
        DRIFT_EMAIL_RECEIVER = config["settings"]["drift_email_receiver"]
        PROJECT_NAME = config["settings"]["project_name"]

        MODEL_PATH = config["model_monitor"]["model_path"]
        REFERENCE_DATA_PATH = config["model_monitor"]["reference_data_path"]
        MODEL_PATH = config["model_monitor"]["model_path"]

        # Take today's data to capture the latest report
        today = datetime.today().strftime("%Y%m%d")
        LATEST_DRIFT_REPORT_PATH = LATEST_DRIFT_REPORT.format(today=today)

        # Read the global column_mapping
        column_mapping = config['model_monitor']['column_mapping'][0]

        logger.info("Getting the model to predict on production data")

        # 2. Get production data
        # As part of the predictions in production, we are expected to use the pre-trained model
        lasso_model = joblib.load(MODEL_PATH)

        # Get the production data (i.e. new data that our model is expected to classify in production)
        # Note: Data has been stored by Ioannis M. for demonstrating purposes and in a real-life case that would come as a schedule job
        df_production_day2 = pd.read_pickle("src/data/assets/df_day2.pickle")

        # 2.1 Call the predict pipeline
        # Make the prediction on the training data (just for demonstrating purposes)
        df_production_day2['prediction'] = lasso_model.predict(df_production_day2)
        df_production_day2['prediction'] = df_production_day2['prediction'].map({1: 'white', 0: 'red'})

        logger.info("Get the ground truth data")

        # 2.2 Get ground truth data
        # Bring in the ground truth data and drop the response because we do not have the ground truth yet for the production data
        # Note: These need to be defined from the config.toml
        from_location = 'your-source-location'
        to_location = 'your-destination-location'
        
        os.system(
            f"hadoop fs -copyToLocal -f {from_location}/df_reference.pickle {to_location}"
        )

        # Read the ground truth data (reference data)
        df_reference = pd.read_pickle(REFERENCE_DATA_PATH)
        df_reference.drop("wine_colour", axis=1, inplace=True)

        # 3. Create the model decay reports
        # Update the column mapping
        column_mapping['target'] = None

        # 3.1 Get model/data drift
        # Generate the model & prediction drift in production (e.g. day 2)
        model_monitor_reports = ModelMonitorReports(df_reference, df_production_day2, column_mapping)
        model_monitor_reports.generate_model_data_drift_report(response_type = 'categorical', report_name = 'poc')

        # 3.3 E-mail if model drift is detected
        monitor_report = MonitorReportReader(LATEST_DRIFT_REPORT_PATH, PROJECT_NAME)
        monitor_report.create_model_drift_table(response_type = 'categorical', p_value_threshold = 1)

        # Send an automated e-mail to capture data drift
        result = monitor_report.send_drift_email_alert(DRIFT_EMAIL_RECEIVER, send_for = 'model_drift')

        # Capture the results of the run
        if result == 0:
            config['messages'] = "Model monitor report is generated :)"
        else:
            config['messages'] = "Error: E-mail was not sent successfully."

        # Store the result of the run
        config['result'] = result
    
    except Exception as e:
        logger.critical(e, exc_info=True)
        config['result'] = -1
        config['message'] = "Error in the model monitor report"

    return {k: config[k] for k in ('result', 'message')}


if __name__ == "__main__":
    res = run_model_decay()
    print(res)
