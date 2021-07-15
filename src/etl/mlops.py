# Import required packages
import pandas as pd
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
        column_mapping['target'] = None

    # If "prediction_name" is not the columns, put None in the prediction (i.e. prediction drift cannot be generated)
    if prediction_name not in df_ref.columns:
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
        report = Dashboard(tabs=[DataDriftTab])
        profile = Profile(sections=[DataDriftProfileSection])

    else:
        AssertionError('Please put either "categorical", "numerical" or "none" on the "response_type" argument!')

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
    custom_report_name = 'reports/model_data_drift_{0}_report_{1}.html'.format(report_name, today)
    custom_profile_name = 'reports/model_data_drift_{0}_report_{1}.json'.format(report_name, today) 
    
    # Export the report as an .html object and .json files
    report.save(custom_report_name)
    
    json_report = profile.json()
    with open(custom_profile_name, 'w') as json_file:
        json.dump(json_report, json_file)



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
    custom_report_name = 'reports/model_performance_{0}_report_{1}.html'.format(report_name, today) 
    profile_report_name = 'reports/model_performance_{0}_report_{1}.json'.format(report_name, today)

    # Export the report as an .html object and .json files
    performance_report.save(custom_report_name)
    
    json_report = profile_report.json()
    with open(profile_report_name, 'w') as json_file:
        json.dump(json_report, json_file)
