import os
import sys

# Allow running this file directly (e.g. `python pipeline/train_pipeline.py`) while still
# supporting package-style imports like `from src...`.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion import DataIngestion
from src.data_transfomation import DataTransformation
from src.model_trainer import ModelTrainer


def main() -> None:
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )

    trainer = ModelTrainer()
    r2 = trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Training completed. R2: {r2}")


if __name__ == "__main__":
    main()
