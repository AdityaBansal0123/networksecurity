from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig
import sys
from networksecurity.entity.config_entity import TrainingPipelineConfig 
if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate data ingestion")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("data initiation completed")
        datavalidation = DataValidation(data_validation_config=DataValidationConfig(training_pipeline_config),data_ingestion_artifact=dataingestionartifact)
        data_validation_artifact = datavalidation.initiate_data_validation()
        logging.info("data validation completed")
        print(data_validation_artifact)
    except Exception as e:
        raise NetworkSecurityException(e,sys)