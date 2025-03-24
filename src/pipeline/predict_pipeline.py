import sys
import os 
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts','model.pkl')
        self.preprocessor_path = os.path.join('artifacts/data','preprocessor.pkl')
        self.model = load_object(self.model_path)
        self.preprocessor = load_object(self.preprocessor_path)
    def predict(self,data):
        try:
            logging.info("Entered the predict method")
            data = self.preprocessor.transform(data)
            return self.model.predict(data)
        except Exception as e:
            raise CustomException(e,sys)
class CustomData:
    def __init__(self,gender:str,race_ethnicity:str,parental_level_of_education:str,
                 lunch:str,test_preparation_course:str,reading_score:int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score

    def get_data_as_data_frame(self):
        try:
            data = {"gender":[self.gender],
                    "race_ethnicity":[self.race_ethnicity],
                    "parental_level_of_education":[self.parental_level_of_education],
                    "lunch":[self.lunch],
                    "test_preparation_course":[self.test_preparation_course],
                    "reading_score":[self.reading_score]}
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e,sys)
        