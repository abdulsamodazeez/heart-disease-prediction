import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.model_evaluation import Acuuracy, F1
from sklearn.base import ClassifierMixin
from catboost.core import CatBoostClassifier

import mlflow
from zenml.client import Client

experiment_tracker= Client().active_stack.experiment_tracker


@step(experiment_tracker= experiment_tracker.name)
def evaluate_model(model:ClassifierMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[
                       Annotated[float, "accuracy"],
                       Annotated[float, "f1"]
                   ]:
    """
        Evaluate the model on the train data

        Args
            df: the ingested trained data 
    """
    try: 
        prediction = model.predict(X_test)
        accuracy= Acuuracy().evaluate_model(y_test, prediction)
        mlflow.log_metric("Accuracy_score", accuracy)


        f1 = F1().evaluate_model(y_test, prediction)
        mlflow.log_metric("F1_score", f1)
        return accuracy, f1
    except Exception as e:
        logging.error("Error in Evaluating the model: {e}")
        return e


