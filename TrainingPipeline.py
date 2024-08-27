from zenml import pipeline,step
from steps.IngestData import ingest_data
from steps.clean_data import clean_data
from steps.model_build import trainmodel
from steps.config import ModelNameConfig
from steps.evaluation import evaluate
import logging

@step(enable_cache=True)
def training_pipeline(data_path:str):
    data=ingest_data(data_path)
    x_train,x_test,y_train,y_test=clean_data(data)
    model=trainmodel(x_train,x_test,y_train,y_test,ModelNameConfig())
    mse,r2=evaluate(model,x_test,y_test)

    logging.info(f"MSE:{mse} and R2_SCORE:{r2}")