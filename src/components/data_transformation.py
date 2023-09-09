from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Data Transformation config
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

# Custom Data Transformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomDataTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Dropping unwanted columns
        cols_to_drop = ['TBG', 'TSH', 'TSH_measured', 'T3_measured',
                        'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured']
        X = X.drop(cols_to_drop, axis=1)

        # Mapping missing values
        X = X.applymap(lambda x: np.nan if x == "?" else x)

        # Mapping dictionary
        mapping_dict = {'t': 1, 'f': 0, 'M': 1, 'F': 0}

        # Apply mapping using a lambda function with applymap()
        X = X.applymap(lambda x: mapping_dict.get(x, x))

        return X

# Data Transformation class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation initiated')
            
            categorical_cols = ['sex', 'on_thyroxine', 'query_on_thyroxine','on_antithyroid_medication', 'thyroid_surgery', 'query_hypothyroid','query_hyperthyroid', 'pregnant', 'sick', 'tumor', 'lithium', 'goitre']
            numerical_cols = ['age','T3', 'TT4', 'T4U', 'FTI']
            
            logging.info('Pipeline initiated')
             
            # Numerical Pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler()),
            ])
            
            # Categorical Pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', MinMaxScaler()),
            ])
            
            # Add the custom transformer for data preprocessing
            data_transformer = CustomDataTransformer()
            
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols),
            ],
            remainder='passthrough')
            
            logging.info('Pipeline completed')
            
            # Include the data transformer in the pipeline steps
            full_pipeline = Pipeline(steps=[
                ('data_transformer', data_transformer),
                ('preprocessor', preprocessor)
            ])
            
            return full_pipeline
                
        except Exception as e:
            logging.error('Error in Data Transformation')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Entered initiate_data_transformation method")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head :\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()
                            
            # Splitting features and target
            value_mapping = {'negative': 0, 'hypothyroid': 1}
            
            input_features_train_df = train_df.drop('Target', axis=1)
            target_feature_train_df = train_df['Target'].map(value_mapping)

            input_feature_test_df = test_df.drop('Target', axis=1)
            target_feature_test_df = test_df['Target'].map(value_mapping)
            
            # Applying the transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Define the oversampling technique
            oversampler = RandomOverSampler(random_state=42)
            
           # Apply oversampling to the preprocessed data
            input_feature_train_arr_oversampled, target_feature_train_arr_oversampled = oversampler.fit_resample(input_feature_train_arr, target_feature_train_df)
            
            # Concatenate target feature using numpy
            train_arr = np.c_[input_feature_train_arr_oversampled, np.array(target_feature_train_arr_oversampled)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Save preprocessing object
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)
            logging.info('Preprocessor pickle created and saved')
            
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        
        except Exception as e:
            logging.error('Exception occurred in initiate_data_transformation.')
            raise CustomException(e, sys)