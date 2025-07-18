import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from catboost import CatBoostRegressor # type: ignore
from sklearn.ensemble import ( # type: ignore
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor # type: ignore
from xgboost import XGBRegressor # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.neighbors import KNeighborsRegressor # type: ignore
from src.utils import save_object,eval_model
from sklearn.metrics import r2_score # type: ignore

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and test input data")
            X_train,Y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "KNearestNeighbour":KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            model_report:dict = eval_model(X_train= X_train,Y_train = Y_train,X_test = X_test,y_test = y_test,models = models)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                logging.info("no best model found")
                raise CustomException("No best model found")
            
            logging.info("Found the best model")
            
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj = best_model    
            )   
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            
            return r2_square
                     
        except Exception as e:
            raise CustomException(e,sys)
    