import os
import sys
import numpy as np  
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import dill
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        import joblib
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)

def evaluate_model(X, y, X_test, y_test, models):
    model_report = {}
    try:
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X, y)
            y_train_pred = model.predict(X)       # Corrected
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            model_report[list(models.keys())[i]] = test_model_score
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise CustomException(e, sys)
    
    return model_report