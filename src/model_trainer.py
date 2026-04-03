import os
import sys

from dataclasses import dataclass
from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }

            param = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                # 'Splitter': ['best', 'random'],
                # 'Max Features': ['sqrt', 'log2'],

                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                # 'Criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                # 'Max Features': ['sqrt', 'log2'],

                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                # 'loss': ['Squared_error', 'huber', 'absolute_error', 'quantile'],
                # 'Criterion': ['squared_error', 'friedman_mse'],
                # 'Max Features': ['auto', 'sqrt', 'log2'],

                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                },
                # 'weights': ['uniform', 'distance'],
                # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],

                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                # 'booster': ['gbtree', 'gblinear'],

                "CatBoosting Regressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'depth': [6, 8, 10]
                },
                # 'iterations': [30, 50, 100],
                # 'depth': [6, 8, 10],
                # 'learning_rate': [.1, .01, .05, .001]
                 
                "AdaBoost Regressor": {
                'learning_rate': [.1, .01, .05, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            # 'loss': ['linear', 'square', 'exponential']
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test,
                                                 y_test=y_test, models=models, param=param)
            
            # model_report values are (test_score, best_params)
            best_model_name, (best_model_score, best_params) = max(
                model_report.items(),
                key=lambda item: item[1][0]
            )

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            
            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
