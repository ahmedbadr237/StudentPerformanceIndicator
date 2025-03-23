import os
import sys 
from src.exception import CustomException
import dill
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as f:
            dill.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train , y_train, X_test , y_test , models):
    try:
        results = pd.DataFrame()
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            #Training set Metrics.
            y_train_pred = model.predict(X_train)
            model_train_MAE = mean_absolute_error(y_train,y_train_pred)
            model_train_r2 = r2_score(y_train,y_train_pred)
            model_train_adjusted_r2 = 1 - ((1 - model_train_r2) * (X_train.shape[0] - 1) / (X_train.shape[0] - X_train.shape[1] - 1))
            #Test set Metrics.
            y_test_pred = model.predict(X_test)
            model_test_MAE = mean_absolute_error(y_test,y_test_pred)
            model_test_r2 = r2_score(y_test,y_test_pred)
            model_test_adjusted_r2 = 1 - ((1 - model_test_r2) * (X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1))
            #Results
            results = pd.concat([results,pd.DataFrame([{
                'Model': list(models.keys())[i],
                'Train MAE': model_train_MAE,
                'Train r2': model_train_r2,
                'Train adjusted r2':model_train_adjusted_r2,
                'test MAE': model_test_MAE,
                'test r2': model_test_r2,
                'test adjusted r2':model_test_adjusted_r2,
            }])],ignore_index=True)
        return results
    except Exception as e:
        raise CustomException(e,sys)

def HyperParameterTuning(cv_models,X_train,y_train):
    '''
    cv models should be a list of tupels , 
    [(model_name_1,model_1,param_1),(model_name_2,model_2,param_2)]
    '''
    best_models={}
    try:
        for model_name , model , param in cv_models:
            search = RandomizedSearchCV(model,param,
            n_iter=10,n_jobs=-1,refit='r2')
            search.fit(X_train,y_train)
            best_params = search.best_params_
            best_model = model.__class__(**best_params)
            best_model.fit(X_train, y_train)
            best_models[model_name] = best_model
        return best_models 
    except Exception as e:
        raise CustomException(e,sys)