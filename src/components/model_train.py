from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomException
import sys
import os
import numpy as np  
import pandas as pd
from dataclasses import dataclass
from catboost import CatBoostClassifier, CatBoostRegressor
from src.utils import save_object, evaluate_model 
from src.utils import evaluate_model 
from sklearn.metrics import r2_score    
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.Model_Trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self, train_array, test_array,preprocessor_path):
        logging.info('Model Trainer Initiated')
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], train_array[:,-1], 
                test_array[:,:-1], test_array[:,-1]
            )
            logging.info('Split train and test data completed')

            models = {
                'LogisticRegression': LogisticRegression(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=3)
            }

            model_report: dict = evaluate_model(
                X=X_train, y=y_train, X_test=X_test, y_test=y_test, models=models
            )

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logging.info(f'Best model found: {best_model_name} with score: {best_model_score}')

            save_object(
                file_path=self.Model_Trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f'Model saved at {self.Model_Trainer_config.trained_model_file_path}')
            
            predicted=best_model.predict(X_test)
            r2=r2_score(y_test, predicted)
            
            return best_model_name, best_model_score ,r2
            
        except Exception as e:
            raise CustomException(e, sys)