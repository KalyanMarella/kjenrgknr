import logging
from numpy.core.multiarray import array as array
from sklearn.metrics import r2_score,mean_squared_error
from abc import ABC,abstractmethod
import numpy as np

class Evaluation(ABC):
    """
    Evaluation of the accuracy of the model
    """
    @abstractmethod
    def calculate_score(self,y_pred:np.ndarray,y_test:np.ndarray):
        """

        Args:
        y_pred : np.array Model prediction
        y_test : np.array Actual Labels

        """
class MSE(Evaluation):

    def calculate_score(self, y_pred: np.ndarray, y_test: np.ndarray)->float:

        try:
            logging.info("Calculating MSE Score")
            mse_score=mean_squared_error(y_test,y_pred)
            return mse_score
        except Exception as e:
            logging.error(f"Error while calculating MSE score {e}")
            raise e
               
class R2_SCORE(Evaluation):

    def calculate_score(self, y_pred: np.ndarray, y_test: np.ndarray)->float:

        try:
            logging.info("Calculating R2 Score")
            r2=r2_score(y_test,y_pred)
            return r2
        except Exception as e:
            logging.error(f"Error while calculating r2 score {e}")
            raise e
        
class RMSE(Evaluation):

    def calculate_score(self, y_pred: np.ndarray, y_test: np.ndarray)->float:

        try:
            logging.info("Calculating RMSE Score")
            rmse_score=mean_squared_error(y_test,y_pred,squared=False)
            return rmse_score
        except Exception as e:
            logging.error(f"Error while calculating RMSE score {e}")
            raise e

