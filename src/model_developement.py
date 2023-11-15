from abc import ABC, abstractmethod
import logging
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

class Model(ABC):
    """
        Abstract class for all model used
    """
    @abstractmethod
    def train(self,X_train, y_train):
        """
        Methods for trainig the model

        Args:
            X_train= training data
            y_train= training target
        Return:
            None
        """
        pass

class CatBoostClassifierModel(Model):
    """
        Class for Linear Regression model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Args:
            X_train= training data
            y_train= training target
        Return:
            None
        """
        try:
            cat= CatBoostClassifier(**kwargs)
            cat.fit(X_train, y_train)
            logging.info("Model training section completed")
            return cat
        except Exception as e:
            logging.error(f"Error in training the model: {e}")
            return e
        

class GradientBoostingClassifierModel(Model):
    """
        Class for Linear Regression model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Args:
            X_train= training data
            y_train= training target
        Return:
            None
        """
        try:
            cat= GradientBoostingClassifier(**kwargs)
            cat.fit(X_train, y_train)
            logging.info("Model training section completed")
            return cat
        except Exception as e:
            logging.error(f"Error in training the model: {e}")
            return e
        

class RandomForestClassifierModel(Model):
    """
        Class for Linear Regression model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Args:
            X_train= training data
            y_train= training target
        Return:
            None
        """
        try:
            cat= RandomForestClassifier(**kwargs)
            cat.fit(X_train, y_train)
            logging.info("Model training section completed")
            return cat
        except Exception as e:
            logging.error(f"Error in training the model: {e}")
            return e
