import os
import pandas as pd
import csv
import shutil
from flask import Flask, render_template, jsonify, request, send_file,redirect,url_for,send_from_directory
from src.exception import CustomException
from src.logger import logging as lg
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipelines.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to my application"


@app.route("/train")
def train_route():
    try:
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)

        return "Training Completed."

    except Exception as e:
        raise CustomException(e, sys)

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    
    try:


        if request.method == 'POST':
            # it is a object of prediction pipeline
            prediction_pipeline = PredictionPipeline(request)
            
            #now we are running this run pipeline method
            prediction_file_detail = prediction_pipeline.run_pipeline()

            lg.info("prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_file_path,
                            download_name= prediction_file_detail.prediction_file_name,
                            as_attachment= True)


        else:
            return render_template('upload_file.html')
    except Exception as e:
        raise CustomException(e,sys)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug= True)