import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


@dataclass
class PredictPipelineConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    model_path: str = os.path.join("artifacts", "model.pkl")


class PredictPipeline:
    def __init__(self, config: PredictPipelineConfig | None = None):
        self.config = config or PredictPipelineConfig()

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        try:
            preprocessor = load_object(self.config.preprocessor_path)
            model = load_object(self.config.model_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return np.asarray(preds)
        except Exception as e:
            raise CustomException(e, sys)


@dataclass
class CustomData:
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float

    def to_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [float(self.reading_score)],
                "writing_score": [float(self.writing_score)],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
