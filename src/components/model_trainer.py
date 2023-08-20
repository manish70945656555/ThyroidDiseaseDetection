#Basis Libraries import
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass

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
            
            models={
            'LogisticRegression':LogisticRegression(),
            'SVC':SVC(),
            'RandomForestClassifier':RandomForestClassifier(),
            'DecisionTreeClassifier':DecisionTreeClassifier(),
            'KNeighborsClassifier':KNeighborsClassifier()
        }   
            # Define hyperparameter grids for each model
            param_grids = {
            'LogisticRegression': {},
            'SVC': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            },
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'DecisionTreeClassifier': {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'KNeighborsClassifier': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        }
            # Performing hyperparameter tuning for each model for improving performace of the models 
            # Initialize variables to keep track of the best model
            best_model_name = None
            best_model_score = 0

            
            best_models = {}
            
            for model_name, model in models.items():
                param_grid = param_grids.get(model_name, {})
                grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                best_models[model_name] = best_model
                
                y_pred = best_model.predict(X_test)
                f1 = f1_score(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Best Hyperparameters for {model_name}: {grid_search.best_params_}")
                print(f"F1-score for {model_name}: {f1}")
                print(f"Accuracy for {model_name}: {accuracy}\n")
    
                # Log the results
                logging.info(f"Best Hyperparameters for {model_name}: {grid_search.best_params_}")
                logging.info(f"F1-score for {model_name}: {f1}")
                logging.info(f"Accuracy for {model_name}: {accuracy}")
    
                # Check if this model has the highest accuracy
                if accuracy > best_model_score:
                    best_model_score = accuracy
                    best_model_name = model_name

            if best_model_name:
                best_model = best_models[best_model_name]
                save_object(
                        file_path=os.path.join('artifacts', f'{best_model_name}_model.pkl'),
                        obj=best_model
                )
                print(f'The best model is {best_model_name} with an accuracy of {best_model_score}')
                
                #The best model found after hyperparameter tuning is RandomForestClassifier with an accuracy of 0.9794628751974723
                
                logging.info(f'The best model is {best_model_name} with an accuracy of {best_model_score}')

            logging.info("Hyperparameter tuning and model evaluation completed.")
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
        
        
