from src.logger import logging  
from src.exception import CustomException
import sys
import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
from src.components.data_transformation import DataTransformation  
from src.components.data_transformation import DataTransformationConfig
from src.components.model_train import ModelTrainer
from src.components.model_train import ModelTrainerConfig
@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("artifacts",'train.csv')
 
    test_data_path=os.path.join("artifacts",'test.csv')
    
    raw_data_path=os.path.join("artifacts",'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method")
        try:
            df=pd.read_csv(r"notebook\data\gemstone.csv")
            logging.info("reading the data and store it in df")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train test data split")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("ingestion of the data done ")
            return(
             self.ingestion_config.train_data_path
            ,self.ingestion_config.test_data_path   
            )
        except Exception as e:
                raise CustomException(e,sys)
            
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    logging.info("Data ingestion completed successfully.")
    data_transformation=DataTransformation()
    train_arr,test_arr,_ =data_transformation.initiate_data_transformation(train_data,test_data)
    logging.info("Data transformation completed successfully.")
    model=ModelTrainer()
    best_model_name, best_model_score, r2 = model.initiate_model_trainer(train_arr, test_arr, DataTransformationConfig().preprocessor_obj_file_path)
    logging.info(f"Best model: {best_model_name} with score: {best_model_score} and R2 score: {r2}")
    logging.info("Model training completed successfully.")
    