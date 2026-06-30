import os
import sys
import pandas as pd
import numpy as np 
from dataclasses import dataclass
from src.exception import CustomeException
from src.log import logging
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')
from src.utils import save_object,evaluate_model

#deciding the path model 
@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
         # we will give the output of data transformation.
        try:
            logging.info("splitting the data into train and test")
            # here we take what is returning in data_transformation
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1], #store all except last column
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model = CatBoostClassifier(
                verbose=False
            )
            model_report= evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,
                                               y_test=y_test,model = model)
            logging.info("model train and evaluted successfully")
            save_object(
                # save the model
                file_path=self.model_trainer_config.train_model_file_path,
                obj=model # convert into pkl file model.pkl
            )
            logging.info("model is saved successfully")

            # to see the predicted o/p for test data
            predicted = model.predict(X_test)
            accuracy = accuracy_score(y_test,predicted)
            return accuracy
        except Exception as e:
            raise CustomeException(e,sys)
            