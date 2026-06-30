import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomeException
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import dill
import pickle

#make function to save obj
def save_object(file_path,obj):
    try:
        dir_path =os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomeException(e,sys)


def evaluate_model(X_train, y_train, X_test, y_test, model):
    try:
        #train model
        model.fit(X_train,y_train)
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Training metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)

        # Testing metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        return {
            "train_accuracy": train_accuracy,
            "train_f1": train_f1,
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall
        }
    

    except Exception as e:
        raise CustomeException(e, sys)