import os
import sys
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow

# Only initialize dagshub if DISABLE_DAGSHUB is not set
if not os.getenv("DISABLE_DAGSHUB"):
    import dagshub
    dagshub.init(repo_owner='AdityaBansal0123', repo_name='networksecurity', mlflow=True)

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self,best_model,classificationmetric):
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score

            mlflow.log_metric("f1_metric",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall_score",recall_score) 
            mlflow.sklearn.log_model(best_model,"model")

    def train_model(self,X_train,y_train,x_test,y_test):
        models = {
            "DecisionTree Classifier": DecisionTreeClassifier(),
            "RandomForest Classifier": RandomForestClassifier(),
            "AdaBoost Classifier": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
        }

        params ={
            "DecisionTree Classifier": {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2', None]
            },

            "RandomForest Classifier": {
                'criterion': ['gini', 'entropy'],
                'n_estimators': [64, 128, 256],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            },

            "Gradient Boosting": {
                'loss': ['log_loss'],
                'learning_rate': [0.1, 0.01, 0.001],
                'n_estimators': [64, 128, 256],
                'subsample': [0.6, 0.8, 1.0],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },

            "AdaBoost Classifier": {
                'learning_rate': [1.0, 0.1, 0.01],
                'n_estimators': [50, 100, 200],
                'algorithm': ['SAMME', 'SAMME.R']
            },
        }

        model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models,params=params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]
        
        y_train_pred=best_model.predict(X_train)
        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
        
        y_test_pred=best_model.predict(x_test)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

        ## Track the experiments with mlflow
        self.track_mlflow(best_model,classification_train_metric)
        self.track_mlflow(best_model,classification_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=Network_Model)
        
        # model pusher
        save_object("final_model/model.pkl", best_model)
        
        ## Model Trainer Artifact 
        model_trainer_artifact=ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_trained_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)