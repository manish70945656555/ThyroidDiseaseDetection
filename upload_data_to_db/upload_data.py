
from pymongo.mongo_client import MongoClient
import pandas pd
import json 

uri = "mongodb+srv://manishkumawat0803:sansad70@cluster0.necirxl.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)

# create database name and collection name
DATABASE_NAME="ThyroidProject"
COLLECTION_NAME="ThyroidDetection"

# read the data as a dataframe
df=pd.read_csv(r"D:\ThyroidDiseaseDetection\notebooks\data\thyroid.csv")

# Convert the data into json
json_record=list(json.loads(df.T.to_json()).values())

#now dump the data into the database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

