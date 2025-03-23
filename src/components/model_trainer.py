import os
import sys
from sklearn.metrics import mean_absolute_error,r2_score
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model,HyperParameterTuning
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('split training and test input data')
            X_train , y_train , X_test , y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            random_forest_params  = {
            'n_estimators': [100, 300, 500],
            'max_depth': [None, 10, 20, 30], 
            'min_samples_split': [2, 5, 10],  
            'min_samples_leaf': [1, 2, 4],  
            'max_features': [ 'sqrt', 'log2'], 
            'bootstrap': [True, False]}
            GBR_param_grid = {
                'n_estimators': [100, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10], 
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.6, 0.8, 1.0],
                'max_features': [ 'sqrt', 'log2']}
            XGBRegressor_param_grid = {
                'n_estimators': [100, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],  
                'max_depth': [3, 5, 7, 10],  
                'min_child_weight': [1, 3, 5], 
                'subsample': [0.6, 0.8, 1.0],  
                'colsample_bytree': [0.6, 0.8, 1.0], 
                'gamma': [0, 0.1, 0.2, 0.5], 
                'reg_alpha': [0, 0.01, 0.1, 1], 
                'reg_lambda': [0, 0.01, 0.1, 1]}
            hyper_parameter_models=[('RandomForest',RandomForestRegressor(),random_forest_params),
                                    ('XGBRegressor',XGBRegressor(),XGBRegressor_param_grid),
                                    ('GradientBoostingRegressor',GradientBoostingRegressor(),GBR_param_grid)]
            best_models = HyperParameterTuning(cv_models=hyper_parameter_models,X_train=X_train,y_train=y_train)

            models = {'RandomForest':best_models['RandomForest'],
                      'DecisionTree':DecisionTreeRegressor(),
                      'Linear':LinearRegression(),
                      'Ridge':Ridge(),
                      'Lasso':Lasso(),
                      'XGBRegressor':best_models['XGBRegressor'],
                      'GradientBoostingRegressor':best_models['GradientBoostingRegressor']}
            evaluation_report = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            best_model_score = evaluation_report['test adjusted r2'].max()
            best_model_name = evaluation_report[evaluation_report['test adjusted r2'] == best_model_score]['Model'].values[0]
            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info('best found model on both training and testing data')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            return best_model_score , best_model_name
        except Exception as e:
            raise CustomException(e,sys)