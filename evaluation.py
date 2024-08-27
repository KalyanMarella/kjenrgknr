from src.Evaluation import R2_SCORE,MSE,RMSE
import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

@step
def evaluate(model:RegressorMixin,x_test:pd.DataFrame,y_test:pd.Series)->Tuple[
    Annotated[float,"mse"],
    Annotated[float,"r2"]
]:

    try:
        y_pred=model.predict(x_test)
        mse=MSE().calculate_score(y_pred,y_test)
        rmse=RMSE().calculate_score(y_pred,y_test)
        r2=R2_SCORE().calculate_score(y_pred,y_test)

        return mse,r2

    except Exception as e:
        logging.error(f"Error while calculating score {e}")
        raise e