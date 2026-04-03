import os
import sys

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        """Creates preprocessing pipelines"""
        try:
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")

            # ✅ STRONG COLUMN CLEANING (FINAL FIX)
            train_df.columns = (
                train_df.columns
                .str.strip()
                .str.lower()
                .str.replace(' ', '_')
                .str.replace('/', '_')
            )

            test_df.columns = (
                test_df.columns
                .str.strip()
                .str.lower()
                .str.replace(' ', '_')
                .str.replace('/', '_')
            )

            logging.info(f"Columns after cleaning: {train_df.columns.tolist()}")

            # ✅ Target column
            target_column_name = 'math_score'

            # ❗ IMPORTANT FIX: Remove target from features
            numerical_columns = ['reading_score', 'writing_score']

            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            # ✅ Validation check
            expected_cols = set(numerical_columns + categorical_columns + [target_column_name])
            missing_cols = expected_cols - set(train_df.columns)

            if missing_cols:
                raise CustomException(f"Missing columns: {missing_cols}", sys)

            # Split input & target
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Data split into input and target")

            # Get preprocessor
            preprocessor_obj = self.get_data_transformer_object(
                numerical_columns,
                categorical_columns
            )

            # Transform data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Data transformation completed")

            # Combine features + target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Preprocessor saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)