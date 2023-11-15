import pandas as pd
import logging
from zenml import step
from src.data_cleaning import DataCleaning, DataPreprocessingStrategy, DataSplittingStrategy
from typing import Tuple
from typing_extensions import Annotated

@step
def clean_dataframe(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"], 
]:
    """
        Data Cleaning section

        Args:
            data: it takes a a dataframe 
        Return:
            it return the splitted data which include:
            X_train, X_test, y_train, y_test
    """
    try:
        data_preprocessing= DataPreprocessingStrategy()
        data_cleaning = DataCleaning(data, data_preprocessing)
        clean_data = data_cleaning.handle_data()

        data_splitting= DataSplittingStrategy()
        data_cleaning= DataCleaning(clean_data, data_splitting)
        X_train, X_test, y_train, y_test= data_cleaning.handle_data()
        logging.info("Data Cleaning Section Completed")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        return e