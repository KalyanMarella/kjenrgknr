import pandas as pd
from zenml import step
from typing_extensions import Annotated
import logging
from typing import Union
from src.CleaningData import DataPreprocessStrategy,DataSplittingStrategy,CleanData

@step
def clean_data(data:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
    try:
        strategy_obj=DataPreprocessStrategy()
        clean_obj=CleanData(data=data,strategy=strategy_obj)
        processed_data=clean_obj.HandleData()

        strategy_obj=DataSplittingStrategy()
        clean_obj=CleanData(data=processed_data,strategy=strategy_obj)
        x_train,x_test,y_train,y_test=clean_obj.HandleData()

        return x_train,x_test,y_train,y_test
    
    except Exception as e:
        logging.error(f"Error while preprocessing the data")
        raise e
