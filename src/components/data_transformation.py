import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

custom_order = [['high school','some college','associate\'s degree','bachelor\'s degree','master\'s degree']]
@dataclass
class DataTransfromationConfig:
    preprocessor_ob_file_path: str = os.path.join('artifacts/data',"preprocessor.pkl") 

class DataTransfomation:
    def __init__(self):
        self.data_transformation_config = DataTransfromationConfig()
    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            numerical_cols = ['reading score']
            ohe_cat_cols = ['gender','race/ethnicity','lunch','test preparation course']
            ordinal_cols = ['parental level of education']
            num_pipeline = Pipeline(
                steps=[("imputer",SimpleImputer(strategy='median')),
                       ('scaler',StandardScaler())]
            )
            cat_pipeline = Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder())])
            logging.info(f"numerical columns :{numerical_cols}")
            logging.info(f"categorical columns :{ohe_cat_cols + ordinal_cols}")
            preprocessor = ColumnTransformer(
                [("numerical_pipline",num_pipeline,numerical_cols),
                 ('cat_pipline',cat_pipeline,ohe_cat_cols),
                 ('ordinal_encoder',OrdinalEncoder(categories=custom_order),ordinal_cols)]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path).drop(columns='writing score')
            test_df = pd.read_csv(test_path).drop(columns='writing score')
            logging.info("read train and test data completed ")
            logging.info("obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math score"
            numerical_cols = ['reading score']
            ohe_cat_cols = ['gender','race/ethnicity','lunch','test preparation course']
            ordinal_cols = ['parental level of education']
            input_feature_train_df = train_df.drop(columns=target_column_name)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=target_column_name)
            target_feature_test_df = test_df[target_column_name]

            logging.info("applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)
            train_array = np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_array,np.array(target_feature_test_df)]
            logging.info('saved preprocessing object')
            
            save_object(file_path=self.data_transformation_config.preprocessor_ob_file_path,obj=preprocessing_obj)
            return(
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_ob_file_path
            )
            logging.info('pkl created successfully')

        except Exception as e:
            raise CustomException(e,sys)