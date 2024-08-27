import pandas as pd
import numpy as np
import logging
from abc import ABC,abstractmethod
from typing import Union
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Idea is to Creating a abstract class so maintain unique configuration of cleaning data
    """
    @abstractmethod
    def HandleData(self):
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Actual Preprocess of the Data
    """
    def HandleData(self,data:pd.DataFrame)->pd.DataFrame:
        try:
            ## cleaning Data
            return data
        except Exception as e:
            logging.error("Error While Preprocessing the data {e}")
            raise e

class DataSplittingStrategy(DataStrategy):
    """
    Splitting the data to training and testing subsets
    """

    def HandleData(self,data:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:

        try:
            x_train,x_test,y_train,y_test=train_test_split(data.drop("target",axis=1),data.target)
            return x_train,x_test,y_train,y_test
        
        except Exception as e:
            logging.error("Error While Splitting Data {e}")
            raise e 
    
class CleanData(DataStrategy):
    """
    Combining of DataPreprocessStrategy and DataSplittingStrategy
    """

    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.data=data
        self.strategy=strategy
    def HandleData(self)->Union[pd.DataFrame,pd.Series]:
        try:
            return self.strategy.HandleData(self.data)
        except Exception as e:
            logging.error("Error While Preprocessing Data {e}")
            raise e

