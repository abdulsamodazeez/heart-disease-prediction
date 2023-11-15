import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from abc import ABC, abstractmethod
from typing import Union

class StrategyData(ABC):
    """
        Abstract class for handling data strategy
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessingStrategy(StrategyData):
    """
        Strategy for handling preprocessing of data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
            data preprocessing
        """
        try:
           # Feature Encoding 

            sex_mapping = {'M': 0, 'F': 1}
            data['Sex'] = data['Sex'].map(sex_mapping)


            chest_mapping = {'ATA': 0, 'NAP': 1, "ASY": 2, "TA": 3}
            data['ChestPainType'] = data['ChestPainType'].map(chest_mapping)


            rest_mapping = {'Normal': 0, 'ST': 1, "LVH": 2}
            data['RestingECG'] = data['RestingECG'].map(rest_mapping)


            exer_mapping = {'N': 0, 'Y': 1}
            data['ExerciseAngina'] = data['ExerciseAngina'].map(exer_mapping)


            st_mapping = {'Up': 0, 'Flat': 1, "Down": 2}
            data['ST_Slope'] = data['ST_Slope'].map(st_mapping)

            return data
        except Exception as e:
            logging.error(f"Erorr in processing data: {e}")
            return e
class DataSplittingStrategy(StrategyData):
    """ 
        Strategy for splitting the data into training set and testing test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X= data.drop("HeartDisease", axis=1)
            y= data["HeartDisease"]

            X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=32)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in splitting data: {e}")
            return e
        
class DataCleaning:
    """
        This class combine both strategy
    """
    def __init__(self,data: pd.DataFrame, strategy: StrategyData) -> None:
        self.df= data
        self.strategy= strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:

        try:
            return self.strategy.handle_data(self.df)
        except Exception as e:
            logging.error(f"Error handling data: {e}")
            return e
