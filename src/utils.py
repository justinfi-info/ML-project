import os
import sys
import pandas as pd
import numpy as np
import dill
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param, cv=3, n_jobs=3, verbose=False):
    try:
        report = {}
        model_names = list(models.keys())
        for i in range(len(model_names)):
            model_name = model_names[i]
            model = models[model_name]
            model_params = param.get(model_name, {})

            gs = GridSearchCV(model, model_params, cv=cv, n_jobs=n_jobs, verbose=verbose, refit=True)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = (test_model_score, gs.best_params_)
            

        return report

    except Exception as e:
        logging.info("Error occurred during model evaluation")
        raise CustomException(e, sys)
