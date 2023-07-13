from sklearn.preprocessing import MinMaxScaler #Handling feature scaling
from sklearn.impute import KNNImputer #Handling missing values
from imblearn.over_sampling import SMOTE,RandomOverSampler,KMeansSMOTE #Handling imbalance data

#pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.impute import SimpleImputer


## Data Transformation config

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
        
## Data Transformationconfig class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
        
    
        
    def get_data_transformation_object(self):
        
        try:
            logging.info('Data transformation initiated')
            # Define the steps for the preprocessor pipeline 
            categorical_cols = ['Target','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication','thyroid_surgery','query_hypothyroid','query_hyperthyroid','pregnant','sick','tumor','lithium','goitre']
            numerical_cols = ['age','T3','TT4','T4U','FTI']
            
            
            logging.info('Pipeline initiated')
             
            #Numerical Pipeline 
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('scaler',MinMaxScaler()),
                ('sampling',RandomOverSampler())
                ]
            )
            
            #Categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                ('sampling',RandomOverSampler())
                ]
            )
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            return preprocessor
            logging.info('Pipeline completed')
                    
        except Exception as e:
            
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        
        logging.info(
            "Entered initiate_data_transforamtion method of Data_Transformation class"
        )
        try:
            #Reading train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head :\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            logging.info('Obtaining preprocessing object')
            
            preprocessing_obj=self.get_data_transformation_object()
            #Defining the columns to drop
           
            independent_cols=['on_thyroxine', 'query_on_thyroxine','on_antithyroid_medication', 'thyroid_surgery', 'query_hypothyroid','query_hyperthyroid', 'pregnant', 'sick', 'tumor', 'lithium', 'goitre']    
           
            #Cleaning data before pipeline
            
            cols_to_drop=['TBG','TSH','TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured']
             #dropping unwanted columns
            train_df=train_df.drop(cols_to_drop,axis=1)
            test_df=test_df.drop(cols_to_drop,axis=1)
            #replacing '?' with nan values
            train_df = train_df.replace('?', np.nan)
            #mapping columns values 
            train_df['Target'] = train_df['Target'].map({'negative': 0, 'hypothyroid': 1})
            train_df.update({col: [1 if val == 't' else 0 for val in train_df[col]] for col in independent_cols})
            train_df['sex'] = train_df['sex'].map({'M': 1, 'F': 0})

            #for test data also doing the same 
            test_df['Target'] = test_df['Target'].map({'negative': 0, 'hypothyroid': 1})
            test_df.update({col: [1 if val == 't' else 0 for val in test_df[col]] for col in independent_cols})
            test_df['sex'] = test_df['sex'].map({'M': 1, 'F': 0})
            
            
            ## features into independent and dependent features
            target_column_name = 'Target'
            drop_columns = [target_column_name]
            
            input_features_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            
            
            #apply the transformation
            
            input_feature_train_arr=preprocessing_obj.fit(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df) 
            
            logging.info('Applying preprocessing object on training and teststing datasets.')
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]   
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] 
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            
            logging.info('Preprocessor pickle in created and saved')   
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                
            )   
        
        except Exception as e:
            logging.info('Exception occured in initiate_data_transformation.')
            
            raise CustomException(e,sys)
           
            