import os
import sys
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Import necessary modules from your project
from src.logger import logging
from src.exception import CustomException

# Data ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')

# Create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()
        
    def export_collection_as_dataframe(self):
        try:
            # Replace with your MongoDB connection details
            client = MongoClient("mongodb+srv://manishkumawat0803:sansad70@cluster0.necirxl.mongodb.net/?retryWrites=true&w=majority")
            db = client["ThyroidProject"]
            collection = db["ThyroidDetection"]
            
            cursor = collection.find()
            data = list(cursor)
            df = pd.DataFrame(data)  # Convert data into a DataFrame
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def export_data_into_feature_store_file_path(self):
        try:
            logging.info("Exporting data from MongoDB")
            raw_file_path = self.ingestion_config.raw_data_path
            os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)
            
            thyroid_data = self.export_collection_as_dataframe()
            logging.info("Saving exported data into feature store file path: {raw_file_path}")
            
            feature_store_file_path = os.path.join('artifacts', 'raw.csv')

            thyroid_data.to_csv(feature_store_file_path, index=False)
            
            return feature_store_file_path
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
            
            # Export data from MongoDB and save it to a CSV file
            feature_store_file_path = self.export_data_into_feature_store_file_path()
            
            logging.info("Got the data from MongoDB")
            
            # Read the exported CSV file using pandas
            df = pd.read_csv(feature_store_file_path)
            
            logging.info("Train test split")
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)
            
            # Save train and test sets as CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error("Error occurred in Data Ingestion config")
            raise CustomException(e, sys)