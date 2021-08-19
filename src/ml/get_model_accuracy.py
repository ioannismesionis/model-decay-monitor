# Import packages
import pandas as pd

# Sklearn
from sklearn.metrics import accuracy_score

# Import logger
import logging
logger = logging.getLogger(__name__)


def get_model_accuracy(y_test, y_pred):
    """
    Get the model accuracy based on the predictions obtained from a model
    and the expected.
    
    Args:
        y_test: (pd.Series): The actual values
        y_pred: (pd.Series): The predictions made by the model
        
    Returns:
        float [0, 1]: Accuracy of the model (e.g. 0.5 --> 50% accuracy)
    """
    logger.info('Getting model accuracy')
    
    return accuracy_score(y_test, y_pred)