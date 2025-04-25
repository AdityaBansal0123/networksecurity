import os
import sys
from networksecurity.cloud.s3_syncer import S3Sync
from networksecurity.constant.training_pipeline import TRAINING_BUCKET_NAME
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
)
from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config = self.training_pipeline_config)
            logging.info("Start data Ingstion")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data Ingestion Completed and artifact")
            return data_ingestion_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config = self.training_pipeline_config)
            logging.info("data validation started")
            data_validation = DataValidation(data_ingestion_artifact = data_ingestion_artifact,data_validation_config = data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation Done and artifact")
            return data_validation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config = self.training_pipeline_config)
            logging.info("Data Transformation Started")
            data_transformation = DataTransformation(data_validation_artifact = data_validation_artifact,data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("data transformation completed with artifact")
            return data_transformation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def start_Model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config = self.training_pipeline_config)
            logging.info("Model Trainer Start")
            model_trainer = ModelTrainer(model_trainer_config = model_trainer_config,data_transformation_artifact = data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("model Trainig Completed")
            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    # Artifact dir to s3 bucket
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_from_s3(folder = self.training_pipeline_config.artifact_dir,aws_bucket_url = aws_bucket_url)
            logging.info(f"Syncing artifact dir to s3 bucket {TRAINING_BUCKET_NAME} completed")
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    # Syncing final_model to S3
    def sync_final_model_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.model_dir,aws_bucket_url = aws_bucket_url)
            logging.info(f"Syncing final model to s3 bucket {TRAINING_BUCKET_NAME} completed")
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def run_pipeline(self):
        data_ingestion_artifact=self.start_data_ingestion()
        data_validation_artifact = self.start_data_validation(data_ingestion_artifact = data_ingestion_artifact)
        data_transformation_artifact = self.start_data_transformation(data_validation_artifact = data_validation_artifact)
        model_trainer_artifact = self.start_Model_trainer(data_transformation_artifact = data_transformation_artifact)
        
        self.sync_artifact_dir_to_s3()
        self.sync_final_model_to_s3()
        logging.info("Training pipeline completed")
        
        return model_trainer_artifact