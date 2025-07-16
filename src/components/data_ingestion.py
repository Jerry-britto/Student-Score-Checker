import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from dataclasses import dataclass

# required for taking input for data ingestion (like train_data path, test_data path,etc.)
@dataclass # using data class not need to use __init__ method to initalize class varialble/properties
class DataIngestionConfig():
    train_data_path: str = os.path.join("artifacts","train.csv") # file path to save training data
    test_data_path: str = os.path.join("artifacts","test.csv") # file path to save test data
    raw_data_path: str = os.path.join("artifacts","data.csv") # file path for rawdata
    
class DataInjestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_injestion(self):
        logging.info("Entered the data ingestion method/component")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Done reading the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train test split initated")
            
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Data Ingestion is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataInjestion()
    obj.initiate_data_injestion()
    