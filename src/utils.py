import os
import sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param=None):
    try:
        report = {}
        param = param or {}

        # Fit each model; if a param grid is provided, do a bounded randomized search.
        for model_name, model in list(models.items()):
            grid = param.get(model_name) or {}

            if grid:
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=grid,
                    n_iter=10,
                    scoring="r2",
                    cv=3,
                    random_state=42,
                    # Keep this single-process to avoid joblib/multiprocessing issues on some Windows setups.
                    n_jobs=1,
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
            else:
                best_model = model.fit(X_train, y_train)

            # Update in-place so the caller can reuse the trained/tuned estimator.
            models[model_name] = best_model

            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        logging.info("Error occurred during model evaluation")
        raise CustomException(e, sys)
