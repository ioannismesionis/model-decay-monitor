# Import packages
import pytest

import os
import sys

import numpy as np

# Change working directory
cwd = os.getcwd()

os.chdir(cwd)
sys.path.append(cwd)

from src.ml.get_model_accuracy import get_model_accuracy


class TestGetModelAccuracy(object):
    @pytest.mark.parametrize('y_test_pred, y_test_actual', [([0, 1, 1], [0, 1, 1]), ([0, 1], [0, 1])])
    def test_on_perfect_fit(self, y_test_pred, y_test_actual):
        """ Testing accuracy for perfect predictions """
        expected_accuracy = 1.0
        actual_accuracy = get_model_accuracy(y_test_pred, y_test_actual)
        
        message = 'Accuracy on perfect fit: Expected {0}, Actual {1}'.format(expected_accuracy, actual_accuracy)
        
        assert expected_accuracy == pytest.approx(actual_accuracy), message
    
    
    @pytest.mark.parametrize('y_test_pred, y_test_actual', [([0, 1, 1], [1, 0, 0]), ([0, 1], [1, 0])])
    def test_on_worst_fit(self, y_test_pred, y_test_actual):
        """ Testing accuracy for worst-case predictions """
        expected_accuracy = 0.0
        actual_accuracy = get_model_accuracy(y_test_pred, y_test_actual)
        
        message = 'Accuracy on imperfect fit: Expected {0}, Actual {1}'.format(expected_accuracy, actual_accuracy)
        
        assert expected_accuracy == pytest.approx(actual_accuracy), message


    @pytest.mark.parametrize('y_test_pred, y_test_actual', [([0, 1, 1, 1], [0, 1, 0, 0]), ([0, 1], [1, 1])])
    def test_on_imperfect_fit(self, y_test_pred, y_test_actual):
        """ Testing accuracy for imperfect predictions """
        expected_accuracy = 0.5
        actual_accuracy = get_model_accuracy(y_test_pred, y_test_actual)
        
        message = 'Accuracy on imperfect fit: Expected {0}, Actual {1}'.format(expected_accuracy, actual_accuracy)
        
        assert expected_accuracy == pytest.approx(actual_accuracy), message























