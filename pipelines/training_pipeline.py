from zenml import pipeline
from steps.ingest_data import ingest_dataframe
from steps.clean_data import clean_dataframe
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    """
        pipeline to collect data, clean data, train and evaluate the model

        Args:
            data_path: the data to the dataset
    """

    df = ingest_dataframe(data_path)
    X_train, X_test, y_train, y_test= clean_dataframe(df)
    model= train_model(X_train, X_test, y_train, y_test)
    accuracyf1, r2_score= evaluate_model(model, X_test, y_test)