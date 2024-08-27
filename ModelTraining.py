from abc import ABC,abstractmethod
import logging
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
class ModelTraining(ABC):
    """
    Training The Model
    """
    @abstractmethod
    def train(self,x_train:pd.DataFrame,y_train:pd.Series):
        """
        Args:
            x_train : Training Data
            y_train : Labels of the data
        Returns:
            Returns trained model 
        """
        pass

class LinearRegressionModel(ModelTraining):

    def train(self, x_train: pd.DataFrame, y_train: pd.Series,**kwargs):
        """
        Args:
            x_train : Training Data
            y_train : Labels of the data
        Returns:
            Returns trained model 
        """
        try:
            model=LinearRegression(**kwargs)
            model.fit(x_train,y_train)
            logging.info("Model Training Completed")
            return model
        except Exception as e:
            logging.error("Error while training the model {e}")
            raise e
