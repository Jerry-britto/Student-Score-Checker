import sys
import pandas as pd # type:ignore
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeLine():
    def __init__(self):
        pass
    
    def predict(self,features):
        model_path = "artifacts/model.pkl"
        preprocessor_path = "artifacts/preprocessor.pkl"
        
        try:
            logging.info("Loading model object")
            model = load_object(file_path = model_path)
            
            logging.info("Loading preprocessor object")
            preprocessor = load_object(file_path = preprocessor_path)
            
            logging.info("Transforming Data")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Predicting Output")
            preds = model.predict(data_scaled)
            
            return preds
        except Exception as e:
            return CustomException(e,sys)
    
class CustomData(): # mapping all input from html to backend
    def __init__(self, gender:str,race_ethnicity:str,parental_level_of_education:str,lunch:str,test_preparation_course:str,reading_score:int,writing_score:int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_frame(self):
        try:
            custom_data_as_dict = {
                "gender":[self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            
            return pd.DataFrame(data=custom_data_as_dict)
        except Exception as e:
            raise CustomException(e,sys)