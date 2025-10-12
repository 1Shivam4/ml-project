import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
            "Linear Regression": LinearRegression(),
            "K Neighbors Regressor":KNeighborsRegressor(),
            "Decision Tree":DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor(),
            "CatBoost  Regressor": CatBoostRegressor(),
            "XGB Regressor": XGBRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor()
            }
            
            params = {
                "Linear Regression":{},
                "K Neighbors Regressor":{
                    "n_neighbors":[5, 7, 9, 11],
                    "weights": ['uniform', 'distance'],
                    "algorithm": ["ball_tree", "kd_tree"]
                },
                "Decision Tree":{
                    "criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "splitter":["best", "random"],
                    "max_features": ["sqrt", "log2"]
                },
                "Random Forest":{
                    "n_estimators": [100, 80, 50, 75],
                    "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    "max_features": ["sqrt", "log2", None]
                },
                "AdaBoost Regressor": {
                    "n_estimators": [100,80, 70, 50],
                    "loss": ["linear", "square", "exponential"]
                },
                "CatBoost  Regressor":{
                    "depth":[6, 8, 10],
                    "learning_rate":[0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                },
                "XGB Regressor":{
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators":[10, 67, 45, 78, 100]
                },
                "Gradient Boosting Regressor":{
                    "n_estimators": [10, 32, 50, 70, 100],
                    "loss": ["squared_error", "absolute_error", "huber", "quantile"],
                    "learning_rate": [0.01, 0.05, 0.1, 0.001]
                }
           }
            
            model_report:dict=evaluate_model(X_train=x_train, y_train=y_train,X_test=X_test, y_test=y_test ,models=models, params=params)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best found model on both training and testing dataset")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)     
            
            predicted = best_model.predict(X_test)       
            r2_scr = r2_score(y_test, predicted)
            
            return r2_scr
        except Exception as e:
            raise CustomException(e, sys)
    