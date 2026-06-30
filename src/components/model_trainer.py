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
            param_grid = {
                            "iterations": [500,700,1000],
                            "learning_rate": [0.03,0.05,0.07,0.1],
                             "depth": [4,5,6,7,8],
                            "l2_leaf_reg": [1, 3, 5, 7, 9],
                            "border_count": [32, 64, 128, 255],
                             "bagging_temperature": [0, 1, 3, 5],
                             "random_strength": [1, 5, 10],
                             "bootstrap_type": ["Bayesian"],
                             #"subsample": [0.6, 0.8, 1.0]
                        }
            model_report= evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,
                                               y_test=y_test,model = model,param = param_grid)
            best_model = model_report['model_']
            logging.info("model train with parameters and evaluted successfully")

            save_object(
                # save the model
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model # convert into pkl file model.pkl
            )
            logging.info(" tuned model is saved successfully")

            # to see the predicted o/p for test data
            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test,predicted)
            return accuracy
        except Exception as e:
            raise CustomeException(e,sys)
            