# Import logger
import joblib
import logging
logger = logging.getLogger('L&L')

# Sklearn
from sklearn.linear_model import LogisticRegression


def get_lasso_model_predictions(X_train, X_test, y_train, save_model = True):
    """
    Train a lasso logistic regression model and gets the predictions
    
    Args:
        X_train (pd.DataFrame): The train data
        X_test (pd.DataFrame): The test set to perform the prediction
        y_train (pd.Series): The response of the train data set
        
    Returns:
        y_pred (pd.Series): The predictions made by the model
    """
    logger.info('Training the Lasso Logistic Regression model')
    
    # Train the model
    lasso = LogisticRegression(penalty = 'l1', solver = 'liblinear')
    lasso.fit(X_train, y_train)
    
    logger.info('Predict on test set')
    
    # Predict
    y_pred = lasso.predict(X_test)
    
    if save_model:
        logger.info('Saving model for future use.')
        joblib.dump(lasso, './src/data/models/lasso_model.sav')
    
    return y_pred