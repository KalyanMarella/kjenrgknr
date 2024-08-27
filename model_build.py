import logging
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from .config import ModelNameConfig
from src.ModelTraining import LinearRegressionModel

@step
def trainmodel(x_train:pd.DataFrame,
               x_test:pd.DataFrame,
               y_train:pd.Series,
               y_test:pd.Series,
               config:ModelNameConfig)->RegressorMixin:
    try:
        if config.modelname=="LinearRegression":
            model=LinearRegressionModel().train(x_train,y_train)
            return model
    except Exception as e:
        logging.error(f"Error while Training the model {e}")
        raise e

