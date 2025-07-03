import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas    as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.Components.data_transformation import DataTransformation
from src.Components.data_transformation import DataTransformationConfig
from src.Components.model_trainer import ModelTrainer





@dataclass
class DataInjectionConfig:
    train_data_path: str = os.path.join('aircrafts', "train.csv")
    test_data_path: str = os.path.join('aircrafts', "test.csv")
    raw_data_path: str = os.path.join('aircrafts', "data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataInjectionConfig()
        
    def initiate_data_ingestion(self):# 4
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('Notebook\data\stud.csv')  #5
            logging.info('Read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #6 saves the raw data to aircrafts file 
            
            logging.info("Train Test intiated")
            train_set,test_set=train_test_split(df, test_size=0.2, random_state=42)# 7  Split the data into train and test
            #80% goes to training and 20% to testing.
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)#saves the train_data to aircrafts file 

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data is completed")
           
            
            return(
                self.ingestion_config.train_data_path, #9 
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=='__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.intiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path))
