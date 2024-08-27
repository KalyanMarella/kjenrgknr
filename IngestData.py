from zenml import step
import pandas as pd
import logging

class IngestData:
    """
    Ingesting Data from the Provided Path
    """
    def __init__(self,data_path:str)->None:
        """
        Initialization of path
        Args:
            data_path : str
        """
        self.data_path=data_path

    def get_data(self)->pd.DataFrame:
        """
        Returning Data
        Args:
            Loading Data from provided path
        """
        logging.info("Reading Data from provided Path")
        data=pd.read_csv(self.data_path)
        return data

@step
def ingest_data(data_path:str)->pd.DataFrame:
    try:
        ingest=IngestData(data_path)
        return ingest.get_data()
    except Exception as e:
        logging.error("Error While Reading the data: {e}")
        raise e 