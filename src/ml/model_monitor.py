# Import packages needed for the classes
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
import configparser as cp

# Email packages
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Import the evidently packages for the dashboard
from evidently.dashboard import Dashboard
from evidently.tabs import (DataDriftTab,
                            NumTargetDriftTab,
                            CatTargetDriftTab,
                            RegressionPerformanceTab,
                            ClassificationPerformanceTab)

# Import the evidently packages for the profile
from evidently.model_profile import Profile
from evidently.profile_sections import (DataDriftProfileSection,
                                        NumTargetDriftProfileSection,
                                        CatTargetDriftProfileSection,
                                        ClassificationPerformanceProfileSection,
                                        RegressionPerformanceProfileSection)

# Create internal logger to be used
logger = logging.getLogger(__name__)



class ModelMonitorReports:
    """

    Model monitor to capture model decay overtime. The class has been built on the "evidently" python package.

    Resources can be found here: https://docs.evidentlyai.com/

    Attributes:
        df_reference (pd.DataFrame): The reference data (ground truth). It should contain at least
                                     the "target" and/or "prediction" columns if report_type is not "none"

        df_current (pd.DataFrame): The data to compare against for underperformance. It should contain the 
                                   exact columns as df_reference.
        
        column_mapping (dictionary): The information needed (e.g. dtypes) in regards to the data in order to generate
                                     the insights into the dashboard.

    Methods:
        update_column_mapping --> None:
            Updates the column_mapping dictionary from the initilisation

        generate_model_data_drift_report --> None:
            Generates and saves the report that shows if any data and/or target/prediction drift has occured overtime

        generate_model_performance_report --> None:
            Generate and saves the report that shows the performance of the model.

    """
    def __init__(self, df_reference, df_current, column_mapping):
        self.df_reference = df_reference
        self.df_current = df_current
        self.column_mapping = column_mapping

    def __repr__(self):
        return (""" Welcome to the Model Monitor package! \n
                    Available methods: \n
                    1. generate_model_data_drift_report()  --> Detects model and/or data drift \n
                    2. generate_model_performance_report() --> Generate performance comparison and metrics \n
                    3. update_column_mapping()             --> Column mapping that powers the information shown in the visualisations. \n \n
                    Enjoy!   
                """)


    def update_column_mapping(self, column_mapping):
        """
        Update the column mapping in case another report needs to be generated

        Args:
            column_mapping (dictionary): The information needed (e.g. dtypes) in regards to the data in order to generate
                                     the insights into the dashboard.

        Returns:
            None
        """
        self.column_mapping = column_mapping


    def generate_model_data_drift_report(self, response_type = 'categorical', report_name = ''):
        """
        Generate the report that shows if any data and/or target/prediction drift has occured overtime
        The report is generated and saved on the current working directory.

        Args:
            response_type (string): The data type of the response (e.g. "categorical", "numerical" or "none")
                                    If none, the report generated is only for data drift.

            report_name (string): Add a custom name for the generated report. This will be on top of the date that is added
                                  by default in the name of the report.

        Returns:
            None: Writes an .html and .json object in the currect working directory.
        """
        # Creating the .copy() objects as we would not like 
        # to manipulate the actual dataframes that are present
        df_ref = self.df_reference.copy()
        df_prod = self.df_current.copy()

        # Capture the columns for prod and reference
        # Note: This is needed for the code to work
        ref_cols_list = sorted(df_ref.columns.tolist())
        prod_cols_list = sorted(df_prod.columns.tolist())

        message = 'Reference and Current data have different columns: Current: {0}, \n Reference: {1}'.format(prod_cols_list, ref_cols_list)
        assert ref_cols_list == prod_cols_list, message

        # Capture the target & prediction names from the column mapping
        # Note: This is needed to check if these variables exist in df_reference to avoid errors for columns that do not exist
        target_name = self.column_mapping['target']
        prediction_name = self.column_mapping['prediction']

        # If both the target_name and prediction_name are missing, the only option is to generate
        # the data drift report. For that option, ensure that " response_type='none' "
        if (target_name not in df_ref.columns) & (prediction_name not in df_ref.columns):
            if response_type != 'none':
                raise AssertionError("""Both 'target' and 'prediction' columns are missing! \n
                                     Set the 'response_type' to 'none' to generate the data drift report.""") 
        
        # If "target_name" is not the columns, put None in the target (i.e. target drift cannot be generated)
        if target_name not in df_ref.columns:
            logger.info('Warning: Target is not available in the data')
            self.column_mapping['target'] = None

        # If "prediction_name" is not the columns, put None in the prediction (i.e. prediction drift cannot be generated)
        if prediction_name not in df_ref.columns:
            logger.info('Warning: Prediction is not available in the data')
            self.column_mapping['prediction'] = None

        # Capture the type of response dtype
        # 1. Categorical variable 
        if response_type == 'categorical':
            report = Dashboard(tabs=[DataDriftTab, CatTargetDriftTab])
            profile = Profile(sections=[DataDriftProfileSection, CatTargetDriftProfileSection])

        # 2. Numerical variable
        elif response_type == 'numerical':
            report = Dashboard(tabs=[DataDriftTab, NumTargetDriftTab])
            profile = Profile(sections=[DataDriftProfileSection, NumTargetDriftProfileSection])

        # 3. None (i.e. capture data drift only)
        elif response_type == "none":
            logger.info('Warning: Only data drift report will be generated')
            report = Dashboard(tabs=[DataDriftTab])
            profile = Profile(sections=[DataDriftProfileSection])
        
        # 4. Wrong input - raise error
        else:
            raise AssertionError('Please put either "categorical", "numerical" or "none" on the "response_type" argument!')

        logger.info('Generating the data and target/prediction drift')

        # Calculate the report of the model and target/predict drift
        report.calculate(df_ref,      # Reference data with target and/or prediction
                         df_prod,     # Current data with target and/or prediction
                         column_mapping=self.column_mapping)

        # Create the json profile report
        profile.calculate(df_ref,      # Reference data with target and/or prediction
                         df_prod,      # Current data with target and/or prediction
                         column_mapping=self.column_mapping)

        # Create the report and profile name to be saved in /mnt
        today = datetime.today().strftime('%Y%m%d')
        custom_report_name = './reports/{today}/model_data_drift_{report_name}_report_{today}.html'.format(report_name = report_name, today = today)
        custom_profile_name = './reports/{today}/model_data_drift_{report_name}_report_{today}.json'.format(report_name = report_name, today = today)

        # Export the report as an .html object and .json files
        # 1. Save as .html
        report.save(custom_report_name)

        # 2. Save as .json
        json_report = profile.json()
        with open(custom_profile_name, 'w') as json_file:
            json.dump(json_report, json_file)

        logger.info('Sucess: Report is generated.')


    def generate_model_performance_report(self, report_type = 'classification', report_name = ''):
        """
        Generate the report that shows the performance of the model.
        The report is generated and saved on the current working directory.

        Requirements:
            1. Column mapping should contain BOTH "target" and "prediction"
            2. df_reference should contain BOTH "target" and "prediction"
            3. df_current should contain BOTH "target" and "prediction"

        Args:
            report_type (string): The type of report to be generated
                                  i.e. "classification" or "regression"

            report_name (string): Add a custom name for the generated report. This will be on top of the date that is added
                                  by default in the name of the report.

        Returns:
            None: Writes an .html and .json object in the currect working directory.
        """
        df_ref = self.df_reference.copy()
        df_prod = self.df_current.copy()

        # Specify the Dashbord to be generated
        # 1. Classification report
        if report_type == 'classification':
            logger.info('Generating the "classification" report')
            performance_report = Dashboard(tabs=[ClassificationPerformanceTab])
            profile_report = Profile(sections=[ClassificationPerformanceProfileSection])

        # 2. Regression report
        elif report_type == 'regression':
            logger.info('Generating the "regression" report')
            performance_report = Dashboard(tabs=[RegressionPerformanceTab])
            profile_report = Profile(sections=[RegressionPerformanceProfileSection])
        
        # 3. Capture error input (e.g. typo) 
        else:
            AssertionError('Please put either "classification" or "regression" on the "report_type" argument!')

        # If the df_prod is not empty (None), ensure that the columns are all present
        if df_prod is not None:
            ref_cols_list = sorted(df_ref.columns.tolist())
            prod_cols_list = sorted(df_prod.columns.tolist())

            assert ref_cols_list == prod_cols_list, 'Reference and prod data have different columns'

            # Capture the target & prediction names from the column mapping
            # This is needed to check if these variables exist in df_ref to avoid errors for columns that do not exist
            target_name = self.column_mapping['target']
            prediction_name = self.column_mapping['prediction']

            if (target_name is None) | (prediction_name is None):
                message = '{0} and {1} should be in the column mapping'.format(target_name, prediction_name)
                AssertionError(message)

            # If "target_name" is not the columns, put None in the target (i.e. target drift cannot be generated)
            if (target_name not in df_ref.columns) | (prediction_name not in df_ref.columns):
                message = '{0} and {1} should be in the df_reference dataframe'.format(target_name, prediction_name)
                AssertionError(message)

            # If "prediction_name" is not the columns, put None in the prediction (i.e. prediction drift cannot be generated)
            if (target_name not in df_prod.columns) | (prediction_name not in df_prod.columns):
                message = '{0} and {1} should be in the df_production dataframe'.format(target_name, prediction_name)
                AssertionError(message)

        # Generating the performance report
        performance_report.calculate(df_ref,    # Reference
                                     df_prod,   # Production
                                     column_mapping=self.column_mapping)

        # Create the json profile report
        profile_report.calculate(df_ref,    # Reference
                                 df_prod,   # Production
                                 column_mapping=self.column_mapping)

        # Create the report name
        today = datetime.today().strftime('%Y%m%d')
        custom_report_name = './reports/{today}/model_performance_{report_name}_report_{today}.html'.format(report_name = report_name, today = today)
        profile_report_name = './reports/{today}/model_performance_{report_name}_report_{today}.json'.format(report_name = report_name, today = today)

        # Export the report as an .html object and .json files
        # 1. Save as .html
        performance_report.save(custom_report_name)

        # 2. Save as .json
        json_report = profile_report.json()
        with open(profile_report_name, 'w') as json_file:
            json.dump(json_report, json_file)

        logger.info('Sucess: Report is generated.')


class MonitorReportReader:
    """
    Read the .json reports generated from the ModelMonitorReports class.

    Attributes:
        json_file_path (str): The full path of the .json file of the latest report to be read
        project_name (string): The project name to be sent to the subject
        
    Methods:
        create_data_drift_table --> pd.DataFrame
            Creates the data drift table from a .json report with the {"column_name", "p_value", "p_value_threshold"} columns

        create_model_drift_table --> pd.DataFrame
            Creates the model drift table from a .json report with the {"column_name", "p_value", "p_value_threshold"} columns
    """
    def __init__(self, json_file_path, project_name = None):

        # Make sure that the arguments are specified as expected
        if (len(json_file_path) == 0) | (project_name is None):
            raise AssertionError("""The 'json_file_path' is empty or project_name is not defined! \n
                                    Please specify the correct arguments.""")

        self.json_file_path = json_file_path
        self.project_name = project_name
        self.project_link = ''
        self.df_target_drift = pd.DataFrame()
        self.df_model_drift = pd.DataFrame()


    def __repr__(self):
        return (""" Welcome to the Monitor Report package! \n
                    Available methods: \n
                    1. create_data_drift_table()  --> Converts a .json report of data drift to a pd.DataFrame \n
                    2. create_model_drift_table() --> Converts a .json report of model drift to a pd.DataFrame \n
                    3. send_drift_email_alert()   --> Sends a personalised e-mail to reciepients about detection of drift \n \n
                    Enjoy!   
                """)

    def _read_json_file(self):
        """
        Read a json file as a json string.

        Args:
            None

        Returns:
            json: Json file read in a string format
        """
        with open(self.json_file_path, 'r') as json_file:
            self.json_string = json.load(json_file)


    def _capture_data_drift(self, df_drift):
        """
        Get the dataframe with the rows that show that drift is detected

        Args:
            df (pd.DataFrame): The data that contains the "drift" column with "Detected" or "Not detected"

        Returns:
            pd.DataFrame: with the rows of interest.
        """
        if 'drift' not in df_drift.columns:
            raise AssertionError('The "drift" column is not present to filter on the rows with drift!')

        # Filter on the rows where drift is detected
        df_drift = df_drift[df_drift['drift'] == 'Detected'].reset_index(drop=True)

        return df_drift
    

    def create_data_drift_table(self, p_value_threshold = 0.05):
        """
        Create the data drift table from a .json report.

        Args:
            p_value_threshold (float): p-value threshold to compare against

        Returns:
            pd.DataFrame: with the columns {"column_name", "p_value", "p_value_threshold"}
        """
        # Create the json_string file
        self._read_json_file()

        # Read the json file
        df_drift = pd.read_json(self.json_string)

        # If the data_drift is present in the report, 
        # capture the data in a pd.DataFrane format
        if 'data_drift' not in df_drift.columns:
            raise AssertionError("The 'data_drift' column should be in the .json file")

        # Create an empty dataframe to store the values
        self.df_data_drift = pd.DataFrame()

        logger.info('Generating the data drift table')

        # Iterate on the pandas data frame metrics (metrics appears as a dict)
        for column, metrics in df_drift.loc['data', 'data_drift']['metrics'].items():
            row = {'column_name': column,
                   'p_value': metrics['p_value'],
                   'p_value_threshold': p_value_threshold}
            self.df_data_drift = self.df_data_drift.append(row, ignore_index=True)

        # Capture if drift is detected in the target/response
        self.df_data_drift['drift'] = np.where(self.df_data_drift['p_value'] < self.df_data_drift['p_value_threshold'],
                                          'Detected',
                                          'Not detected')


    def create_model_drift_table(self, response_type = 'categorical', p_value_threshold = 0.05):
        """
        Create the model drift table from a .json report.

        Args:
            response_type (str): Data type of the response & prediction ("categorical" of "numerical")
            p_value_threshold (float): p-value threshold to compare against

        Returns:
            pd.DataFrame: with the columns {"column_name", "p_value", "p_value_threshold"}
        """
        # Create the json_string file
        self._read_json_file()

        # Read the json string
        df_drift = pd.read_json(self.json_string)

        logger.info('Generating the model drift table')

        if response_type == 'categorical':
            # The 'cat_target_drift' should be present in the .json report 
            assert 'cat_target_drift' in df_drift.columns, 'The "cat_target_drift" column is not present in the json file'

            # Create the table with the target/prediction drift
            self.df_target_drift = pd.DataFrame(df_drift.loc['data', 'cat_target_drift']['metrics'], index=[0])

        elif response_type == 'numerical':
            # The 'num_target_drift' should be present in the .json report
            assert 'num_target_drift' in df_drift.columns, 'The "num_target_drift" column is not present in the json file'

            # Create the table with the target/prediction drift
            self.df_target_drift = pd.DataFrame(df_drift.loc['data', 'num_target_drift']['metrics'], index=[0])

        else:
            raise AssertionError('Please put either "categorical" or "numerical" on the "response_type" argument!')

        # If true, that means that we have prediction & target drift
        if len(self.df_target_drift.columns) > 3:

            # Take the first 3 & 3 last columns and stack them
            # First 3: Prediction drift, Last 3: Target drift
            p_values = np.vstack((self.df_target_drift.iloc[:, :3], self.df_target_drift.iloc[:, -3:]))
            self.df_target_drift = pd.DataFrame(p_values, columns = ['drift_column', 'dtype', 'p_value'])

        else:
            # Just rename the columns for consistency
            self.df_target_drift.columns = ['drift_column', 'dtype', 'p_value']

        # Generate the p_value threshold column
        self.df_target_drift['p_value_threshold'] = p_value_threshold

        # Capture if drift is detected in the target/response
        self.df_target_drift['drift'] = np.where(self.df_target_drift['p_value'] < self.df_target_drift['p_value_threshold'],
                                                 'Detected',
                                                 'Not detected')


    def send_drift_email_alert(self, email_recipients, send_for = 'data_drift'):
        '''
        Send pre-specified e-mail to recipients e-mail addresses.
        
        Args:
            email_recipients (list): E-mail addresses that an e-mail will be sent
            project_name (string): The project name to be sent to the subject
            df_drift_table (pd.DataFrame): The table that contains the informatin of drift or not
        
        Returns:
            None
        '''
        try:
            # Ensure that the data_drift dataframe is not empty
            if send_for == 'data_drift':

                # If the data is empty then we should stop the execution
                if self.df_data_drift.empty:
                    logger.critical('The Dataframe for data drift is empty!')

                    return -1

                df_drift_table = self.df_data_drift.copy()

            # Ensure that the model_drift dataframe is not empty
            elif send_for == 'model_drift':

                # If the data is empty then we should stop the execution
                if self.df_target_drift.empty:
                    logger.critical('The Dataframe for model drift is empty!')

                    return -1

                df_drift_table = self.df_target_drift.copy()

            else:
                raise AssertionError('Please put "data_drift" or "model_drift" in the send_for argument!')
            
            # Check if drift is detected
            df_drift_table = self._capture_data_drift(df_drift_table)

            # If no drift, then return -2
            if df_drift_table.empty:
                logger.info('No drift detected!')

                return -2

            logger.info('Drift is detected - contacting Data Science & Analytics team!')

            msg = MIMEMultipart('alternative')
            
            # Update the subject of the e-mail
            msg['Subject'] = 'Drift Detected in project: {}'.format(self.project_name)
            
            # Capture the path of the dashboard
            # Note: It should be the exact same name with changed extension (.html is expected)
            report_path = self.json_file_path.replace('.json', '.html').replace('/mnt', '')
            
            # Create the project link
            self.project_link = 'https://domino.europe.easyjet.local/u/{0}/view{1}'.format(self.project_name, report_path)
            
            # Update the body of the e-mail as a table with the result and the corresponding message
            html = """\
                    <html>
                      <head></head>
                      <body>
                        {0} <br>
                            Please check the reports at the following link: <br>
                        {1}
                      </body>
                    </html>
                    """.format(df_drift_table.to_html(), self.project_link)
            msg.attach(MIMEText(html, 'html'))
            
            # Specify the server
            conn = smtplib.SMTP('smtp.gmail.com', 587)
            conn.starttls()

            # Login to e-mail and send message
            conn.login('my_email@gmail.com', 'my_password123')
            conn.sendmail('my_email@gmail.com', email_recipients, msg.as_string())

            # Close the connection
            conn.quit()
            
            return 0

        except Exception as e:
            raise Exception(f"Could not sent the e-mail. Error: {e}")