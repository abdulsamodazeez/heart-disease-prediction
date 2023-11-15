from zenml import step
import logging
import pandas as pd
from src.model_developement import (CatBoostClassifierModel, 
                                    GradientBoostingClassifierModel,
                                    RandomForestClassifierModel)
from sklearn.base import ClassifierMixin
from .config import ModelConfig
from catboost.core import CatBoostClassifier
import mlflow
from zenml.client import Client

experiment_tracker= Client().active_stack.experiment_tracker


@step(experiment_tracker= experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelConfig
) -> ClassifierMixin:
    """
        Train the model on the clean ingested data 
    """
    try:
        model= None
        if "RandomForestClassifier" in config.model_name:
            mlflow.sklearn.autolog()
            model = RandomForestClassifierModel().train(X_train, y_train)
            return model
        else:
            raise ValueError(f"Model {config.model_name} not listed")
    except Exception as e:
        logging.error(f"Error training the model: {e}")
        return e