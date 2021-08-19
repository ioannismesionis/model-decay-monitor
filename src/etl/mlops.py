# Email packages
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Import required packages
import pandas as pd
import numpy as np
from datetime import datetime
import json

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

import logging
logger = logging.getLogger('L&L')


def create_column_mapping(df_ref, target = None, prediction = None):
    """
    Generate the column mapping dictionary needed for the data & model drift.
    
    Args:
        df_ref (pd.DataFrame): Reference data (ground truth)
        target (str): The name of the targer/response variable
        prediction (str): The name of the column where the predictions are stored in the df_ref
        
    Returns:
        dictionary: that contains the data types of the columns and the mapping information
                    for the evidently Dashboard() class 
    """
    df_ref = df_ref.copy()
    column_mapping = {}

    logger.debug('Create the column_mapping dictionary')

    # Get the dtypes of python available
    num_types = ['float64', 'float32', 'int64', 'int32']
    categorical_types = ['object']
    datetime_types = ['datetime64[ns]']
    
    # Drop the predictions and/or target column if present
    if target:
        df_ref = df_ref.drop(target, axis=1)
    
    if prediction:
        df_ref = df_ref.drop(prediction, axis=1)

    # Get numeric/categorical and datetime columns
    numerical_features = df_ref.select_dtypes(include=num_types).columns.to_list()
    categorical_features = df_ref.select_dtypes(include=categorical_types).columns.to_list()
    datetime_features = df_ref.select_dtypes(include=datetime_types).columns.to_list()

    # Get the target and prediction
    column_mapping['target'] = target
    column_mapping['prediction'] = prediction

    # Get the feature names of the column mapping
    column_mapping['datetime'] = datetime_features
    column_mapping['numerical_features'] = numerical_features
    column_mapping['categorical_features'] = categorical_features
    
    return column_mapping


def generate_model_data_drift_report(df_ref, df_prod, column_mapping = None, response_type = 'categorical', report_name = ''):
    """
    Generate the report that shows if any data and/or target/prediction drift has occured overtime
    The report is generated and saved on the current working directory.

    Args:
        df_ref (pd.DataFrame): The reference data (ground truth). It should contain at least
                               the "target" and/or "prediction" columns if report_type is not "none"

        df_prod (pd.DataFrame): The data to compare against for underperformance. It should contain the 
                                exact columns as df_ref.

        column_mapping (dictionary): The information needed (e.g. dtypes) in regards to the data in order to generate
                                     the insights into the dashboard.

        response_type (string): The data type of the response (e.g. "categorical", "numerical" or "none")
                                If none, the report generated is only for data drift.

        report_name (string): Add a custom name for the generated report. This will be on top of the date that is added
                              by default in the name of the report.

    Returns:
        None: Writes an .html and .json object in the currect working directory.
    """
    df_ref = df_ref.copy()
    df_prod = df_prod.copy()

    # Capture the columns for prod and reference
    # Note: This is needed for the code to work
    ref_cols_list = sorted(df_ref.columns.tolist())
    prod_cols_list = sorted(df_prod.columns.tolist())

    assert ref_cols_list == prod_cols_list, 'Reference and prod data have different columns'

    # Capture the target & prediction names from the column mapping
    # This is needed to check if these variables exist in df_ref to avoid errors for columns that do not exist
    target_name = column_mapping['target']
    prediction_name = column_mapping['prediction']

    # If "target_name" is not the columns, put None in the target (i.e. target drift cannot be generated)
    if target_name not in df_ref.columns:
        logger.debug('Warning: Target is not available in the data')
        column_mapping['target'] = None

    # If "prediction_name" is not the columns, put None in the prediction (i.e. prediction drift cannot be generated)
    if prediction_name not in df_ref.columns:
        logger.debug('Warning: Prediction is not available in the data')
        column_mapping['prediction'] = None

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
        logger.debug('Warning: Only data drift report will be generated')
        report = Dashboard(tabs=[DataDriftTab])
        profile = Profile(sections=[DataDriftProfileSection])

    else:
        raise AssertionError('Please put either "categorical", "numerical" or "none" on the "response_type" argument!')

    logger.debug('Generating the data and target/prediction drift')

    # Calculate the report of the model and target/predict drift
    report.calculate(df_ref,      # Reference data with target and/or prediction
                     df_prod,     # Current data with target and/or prediction
                     column_mapping=column_mapping)

    # Create the json profile report
    profile.calculate(df_ref,      # Reference data with target and/or prediction
                     df_prod,      # Current data with target and/or prediction
                     column_mapping=column_mapping)

    # Create the report and profile name
    today = datetime.today().strftime('%Y%m%d')
    custom_report_name = '/mnt/reports/{today}/model_data_drift_{report_name}_report_{today}.html'.format(report_name = report_name, today = today)
    custom_profile_name = '/mnt/reports/{today}/model_data_drift_{report_name}_report_{today}.json'.format(report_name = report_name, today = today)
    
    # Export the report as an .html object and .json files
    report.save(custom_report_name)
    
    json_report = profile.json()
    with open(custom_profile_name, 'w') as json_file:
        json.dump(json_report, json_file)

    logger.debug('Sucess: Report is generated.')



def generate_model_performance_report(df_ref, df_prod, column_mapping = None, report_type = 'classification', report_name = ''):
    """
    Generate the report that shows the performance of the model.
    The report is generated and saved on the current working directory.

    Args:
        df_ref (pd.DataFrame): The reference data (ground truth). It should contain at least
                               the "target" and/or "prediction" columns if report_type is not "none"

        df_prod (pd.DataFrame): The data to compare against for underperformance. It should contain the 
                                exact columns as df_ref.
        
        column_mapping (dictionary): The information needed (e.g. dtypes) in regards to the data in order to generate
                                     the insights into the dashboard.

        report_type (string): The type of report to be generated (i.e. "classification" or "regression")

        report_name (string): Add a custom name for the generated report. This will be on top of the date that is added
                              by default in the name of the report.

    Returns:
        None: Writes an .html and .json object in the currect working directory.
    """
    df_ref = df_ref.copy()
    df_prod = df_prod.copy()
    
    # Specify the Dashbord to be generated
    # 1. Classification report
    if report_type == 'classification':
        logger.debug('Generating the "classification" report')
        performance_report = Dashboard(tabs=[ClassificationPerformanceTab])
        profile_report = Profile(sections=[ClassificationPerformanceProfileSection])

    # 2. Regression report
    elif report_type == 'regression':
        logger.debug('Generating the "regression" report')
        performance_report = Dashboard(tabs=[RegressionPerformanceTab])
        profile_report = Profile(sections=[RegressionPerformanceProfileSection])

    else:
        AssertionError('Please put either "classification" or "regression" on the "report_type" argument!')

    # If the df_prod is not empty (None), ensure that the columns are all present
    if df_prod is not None:
        ref_cols_list = sorted(df_ref.columns.tolist())
        prod_cols_list = sorted(df_prod.columns.tolist())

        assert ref_cols_list == prod_cols_list, 'Reference and prod data have different columns'

        # Capture the target & prediction names from the column mapping
        # This is needed to check if these variables exist in df_ref to avoid errors for columns that do not exist
        target_name = column_mapping['target']
        prediction_name = column_mapping['prediction']

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
                                 column_mapping=column_mapping)

    # Create the json profile report
    profile_report.calculate(df_ref,    # Reference
                             df_prod,   # Production
                             column_mapping=column_mapping)

    # Create the report name
    today = datetime.today().strftime('%Y%m%d')
    custom_report_name = '/mnt/reports/{today}/model_performance_{report_name}_report_{today}.html'.format(report_name = report_name, today = today)
    profile_report_name = '/mnt/reports/{today}/model_performance_{report_name}_report_{today}.json'.format(report_name = report_name, today = today)

    # Export the report as an .html object and .json files
    performance_report.save(custom_report_name)
    
    json_report = profile_report.json()
    with open(profile_report_name, 'w') as json_file:
        json.dump(json_report, json_file)

    logger.debug('Sucess: Report is generated.')


def create_data_drift_table(json_file, p_value_threshold = 0.05):
    """
    Create the data drift table from a .json report.

    Args:
        json_file (str): Json file in the string format
        p_value_threshold (float): p-value threshold to compare against
        
    Returns:
        pd.DataFrame: with the columns {"column_name", "p_value", "p_value_threshold"}
    """
    # Read the json file
    df_drift = pd.read_json(json_file)
    
    # If the data_drift is present in the report, 
    # capture the data in a pd.DataFrane format
    if 'data_drift' not in df_drift.columns:
        raise AssertionError("The 'data_drift' column should be in the .json file")
    
    # Create an empty dataframe to store the values
    df_data_drift = pd.DataFrame()

    for column, metrics in df_drift.loc['data', 'data_drift']['metrics'].items():
        row = {'column_name': column,
            'p_value': metrics['p_value'],
            'p_value_threshold': p_value_threshold}
        df_data_drift = df_data_drift.append(row, ignore_index=True)
    
    # Capture if drift is detected in the target/response
    df_data_drift['drift'] = np.where(df_data_drift['p_value'] < df_data_drift['p_value_threshold'],
                                      'Detected',
                                      'Not detected')

    return df_data_drift


def create_model_drift_table(json_file, response_type = 'categorical', p_value_threshold = 0.05):
    """
    Create the model drift table from a .json report.

    Args:
        json_file (str): Json file in the string format
        response_type (str): Data type of the response & prediction ("categorical" of "numerical")
        p_value_threshold (float): p-value threshold to compare against
        
    Returns:
        pd.DataFrame: with the columns {"column_name", "p_value", "p_value_threshold"}
    """
    # Read the json file
    df_drift = pd.read_json(json_file)
    
    if response_type == 'categorical':
        # The 'cat_target_drift' should be present in the .json report 
        assert 'cat_target_drift' in df_drift.columns, 'The "cat_target_drift" column is not present in the json file'
        
        # Create the table with the target/prediction drift
        df_target_drift = pd.DataFrame(df_drift.loc['data', 'cat_target_drift']['metrics'], index=[0])
        
    elif response_type == 'numerical':
        # The 'num_target_drift' should be present in the .json report
        assert 'num_target_drift' in df_drift.columns, 'The "num_target_drift" column is not present in the json file'
    
        # Create the table with the target/prediction drift
        df_target_drift = pd.DataFrame(df_drift.loc['data', 'num_target_drift']['metrics'], index=[0])
        
    else:
        raise AssertionError('Please put either "categorical" or "numerical" on the "response_type" argument!')
        
    # If true, that means that we have prediction & target drift
    if len(df_target_drift.columns) > 3:
        
        # Take the first 3 & 3 last columns and stack them
        # First 3: Prediction drift, Last 3: Target drift
        p_values = np.vstack((df_target_drift.iloc[:, :3], df_target_drift.iloc[:, -3:]))
        df_target_drift = pd.DataFrame(p_values, columns = ['drift_column', 'dtype', 'p_value'])

    else:
        # Just rename the columns for consistency
        df_target_drift.columns = ['drift_column', 'dtype', 'p_value']
        
    # Generate the p_value threshold column
    df_target_drift['p_value_threshold'] = p_value_threshold
    
    # Capture if drift is detected in the target/response
    df_target_drift['drift'] = np.where(df_target_drift['p_value'] < df_target_drift['p_value_threshold'],
                                        'Detected',
                                        'Not detected')
    
    return df_target_drift


def read_json_file(file_path):
    """
    Read a json file as a json string.

    Args:
        file_path (str): The path of the .json file to read

    Returns:
        json: Json file read in a string format
    """
    logger.debug('Reading the .json file.')

    with open(file_path, 'r') as json_file:
        json_string = json.load(json_file)

    return json_string


def email_generator(email_recipients, project_name, report_path, df_drift_table):
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
        # Convert the result to a pandas dataframe
        df_drift_table = df_drift_table.copy()
        
        msg = MIMEMultipart('alternative')
        
        # Update the subject of the e-mail
        msg['Subject'] = 'Drift Detected in project: {}'.format(project_name)

        # Capture the path of the dashboard
        # Note: It should be the exact same name with changed extension (.html is expected)
        report_path = report_path.replace('.json', '.html').replace('/mnt', '')

        # Create the project link
        project_link = 'https://domino.europe.easyjet.local/u/{0}/view{1}'.format(project_name, report_path)
        
        # Update the body of the e-mail as a table with the result and the corresponding message
        html = """\
                <html>
                  <head></head>
                  <body>
                    {0} <br>

                    Please check the reports in the following link: <br>
                    {1}
                  </body>
                </html>
                """.format(df_drift_table.to_html(), project_link)

        msg.attach(MIMEText(html, 'html'))
        
        # Specify the server
        server = smtplib.SMTP('smtp.europe.easyjet.local', 25)
        server.ehlo
        
        # Iterate on the list that holds the e-mail addresses
        for email in email_recipients:
            server.sendmail('domino_script@easyjet.com', [email], msg.as_string())
            
        # Close the connection
        server.close()
        
    except Exception as e:
        raise Exception(f"Could not sent the e-mail. Error: {e}")


def string_to_list(string, split = ',', allow_empty_string=False): 
    '''
    Converts a string to a list using a ',' as the delimiter string.
    
    Args:
        string (string): a string input delimited by commas where the split of the string is required for the final list output.
        split (string): type of split to be used (currently supporting "," and "~")
        allow_empty_string (bool, default False): option to remove empty strings from the returned list
    
    Returns:
        list: from the string that is used as an input.
    '''
    if split == ',':
        li = list(string.split(",")) 
    elif split == '~':
        li = list(string.split("~"))
        
    if not allow_empty_string:
        li = [s for s in li if s] # removes empty strings
        
    return li 
