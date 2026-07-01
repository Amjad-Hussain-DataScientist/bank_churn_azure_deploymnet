import sys
import pandas as pd
from src.exception import CustomeException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    # here we will make another function which make prediction
    def predict(self, features):
        try:
            # here we bring pkl file from artifact giving path
            model_path = r'artifacts\model.pkl'
            preprocessor_path = r'artifacts\preprocessor.pkl'
            # now loading the model_path using load_object from utils 
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            # once we load the pkl of prefrocessor then we transform the feature
            data_scaled = preprocessor.transform(features)
            # after transformation model do the prediction
            # Get probability of class 1
            probabilities = model.predict_proba(data_scaled)[:, 1]
            THRESHOLD = 0.35
            preds = (probabilities >= THRESHOLD).astype(int)
            return preds 
        except Exception as e:
            raise CustomeException(e,sys)



class CustomData:
    def __init__(self,
                 credit_score:int,
                 country:str,
                 gender:str,
                 age:int,
                 tenure:int,
                 balance:float,
                 products_number:int,
                 credit_card:int,
                 active_member:int,
                 estimated_salary:float
                 ):
        # now creating variable using self
        self.credit_score=credit_score
        self.country = country
        self.gender = gender
        self.age = age
        self.tenure = tenure
        self.balance = balance
        self. products_number =  products_number
        self.credit_card = credit_card
        self.active_member = active_member
        self.estimated_salary = estimated_salary
    #now making function which return all data in the form of datafram
    def get_data_as_data_frame(self):
      try:
        custom_data_input_dict = {
           "credit_score":[self.credit_score],
           "country":[self.country], 
           "gender":[self.gender],
           "age":[self.age],
           "tenure": [self.tenure],
           "balance":[self.balance], 
           "products_number":[self. products_number],
           "credit_card": [self.credit_card],
           "active_member": [self.active_member],
           "estimated_salary" : [self.estimated_salary]
           

        }
        # now change custom_data_input_dict intp df
        return pd.DataFrame(custom_data_input_dict)
      except Exception as e:
        raise CustomeException(e,sys)