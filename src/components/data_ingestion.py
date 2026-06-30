import os 
import sys
import pandas as pd
from src.exception import CustomeException
from src.log import logging
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
     self.ingestion_config = DataIngestionConfig()

    def data_ingestion_initiate(self):
        logging.info("data ingestion starts")
        try:
           df = pd.read_csv(r'notebook\Data\churn.csv')
           logging.info('Reading Data as datafram')

           # making path and directory for train data
           os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
           # now for raw data
           df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

           logging.info('Train test split initiated')
           train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

           train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            # to save test set into artifact
           test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
           logging.info('Ingestion of data is completed')
           return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)
        except Exception as e:
           raise CustomeException(e,sys)

#initiating dataingestion
if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.data_ingestion_initiate()
    #to check trnsformation we make obj of class and call all function that we have
    data_transfromation = DataTransformation()
    data_transfromation.initiate_data_transformation(train_data,test_data)
