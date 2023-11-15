import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from abc import ABC, abstractmethod

class Evaluation(ABC):
    
    @abstractmethod
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        """
            model evaluation strategy
            Args:
                y_true: the actual output
                y_pred: the predicted output
        """
        pass

class Acuuracy(Evaluation):
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        try: 
            acc_score = accuracy_score(y_true, y_pred)
            logging.info(f"The Accuracy_Score is: {acc_score}")
            return acc_score
        except Exception as e:
            logging.error(f"Error in evaluating model {e}")
            return e
        
class F1(Evaluation):

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        try: 
            f1 = f1_score(y_true, y_pred)
            logging.info(f"The F1_score is: {f1}")
            return f1
        except Exception as e:
            logging.error(f"Error in evaluating model {e}")
            return e
        
