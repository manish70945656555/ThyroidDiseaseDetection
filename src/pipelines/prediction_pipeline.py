import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
from flask import request

from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = 'predictions'
    prediction_file_name: str = 'predicted_file.csv'
    model_file_path: str = os.path.join('artifacts', "RandomForestClassifier_model.pkl")
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    prediction_file_path: str = os.path.join(prediction_output_dirname, prediction_file_name)

class PredictionPipeline:
    def __init__(self, request: request):
        self.request = request
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def save_input_files(self) -> str:
        """
        Method Name : save_input_files
        Description : This method saves the input file to the prediction artifacts directory
        Output : input dataframe
        On Failure : write an exception log and then raise an exception
        Version : 1.2
        Revisions : moved setup to cloud
        """
        try:
            # creating the file
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)

            input_csv_file.save(pred_file_path)

            return pred_file_path

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            model = load_object(self.prediction_pipeline_config.model_file_path)
            preprocessor = load_object(file_path=self.prediction_pipeline_config.preprocessor_path)

            transformed_x = preprocessor.transform(features)

            preds = model.predict(transformed_x)

            return preds

        except Exception as e:
            raise CustomException(e, sys)

    def get_predicted_dataframe(self, input_dataframe_path: pd.DataFrame):
        """
        Method Name :   get_predicted_dataframe
        Description :   this method returns the dataframe with a new column containing predictions
        Output      :   predicted dataframe
        On Failure  :   Write an exception log and then raise an exception
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            prediction_column_name: str = "Target"
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)

            predictions = self.predict(input_dataframe)
            
            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname, exist_ok=True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index=False)
            
            
            logging.info("Prediction file created and saved ")
            
            logging.info("predictions completed.")

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)
            path="predictions/predicted_file.csv"
            
            return path

        except Exception as e:
            raise CustomException(e, sys)
