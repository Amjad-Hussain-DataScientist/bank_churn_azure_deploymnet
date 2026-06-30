import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from src.exception import CustomeException
from src.log import logging
from src.utils import save_object
from dataclasses import dataclass
from imblearn.over_sampling import SMOTE

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation_object(self):
        try:
            numerical_columns = [
                'credit_score', 'age', 'tenure', 'balance', 'products_number',
                'credit_card', 'active_member', 'estimated_salary'
            ]

            categorical_columns = [
                'country','gender'
            ]
            # creating pipline
            # 1. for numerical 
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            # 2. catg pipline
            cat_pipeline = Pipeline(
                steps =[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoding',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))

                ]
            )

            logging.info('numerical scaling and missing is completed')
            logging.info('categorical encoded and missing is completed')

            # using ColumnTransform
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomeException(e,sys)
    
    # starting data transformation
    def initiate_data_transformation(self,train_path,test_path):
        try:
            #reading data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read the train and test data")
            logging.info('getting preprocessing object')
            # droping unwanted column 
            drop_column = ['customer_id']
            train_df = train_df.drop(columns=drop_column)
            test_df = test_df.drop(columns=drop_column)

            preprocessing_obj = self.get_data_transformation_object()

            #target column
            target_column_name = 'churn'

            # defining the input features for training data
            input_feature_train_df = train_df.drop(columns = ['churn'])
            target_feature_train_df = train_df[target_column_name]

            # same for test data
            input_feature_test_df = test_df.drop(columns = ['churn'])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Apply preprocessing object on train and test data")
            input_feature_train_arr =preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Apply SMOTE ONLY on training data
            smote = SMOTE(random_state=42)
            input_feature_train_arr, target_feature_train_df = smote.fit_resample(
                             input_feature_train_arr,
                            target_feature_train_df
                )

            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]
            logging.info('save preprocessing object')

            # save the pickle file 
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            # return 3 variable
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomeException(e,sys)
