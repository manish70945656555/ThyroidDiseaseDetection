#Basis Libraries import
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
import pandas as pd
import numpy as np

import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent from train and test data ')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
        

            
            #we dont perform any type of hyperparameter tuning because our algorithm random forest classifier already giving highest accuracy
            model=RandomForestClassifier()
            model.fit(X_train, y_train)
            
            save_object(
                file_path=os.path.join('artifacts','model.pkl'),
                obj=model
            )
            
    
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"F1-score for {model}: {f1}")
            print(f"Accuracy for {model}: {accuracy}\n")
    
            # Log the results
            logging.info(f"F1-score for {model}: {f1}")
            logging.info(f"Accuracy for {model}: {accuracy}")
    
            
            logging.info("Model Training completed and model.pkl created and saved in artifacts")
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
        
        
