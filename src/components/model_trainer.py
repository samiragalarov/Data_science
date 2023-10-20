import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try: 
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
              train_array[:,:-1],
              train_array[:,-1],
              test_array[:,:-1],
              test_array[:,-1]
            )


            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
                "Bernoulli Naive Bayes": BernoulliNB(),
                "Support Vector Classifier": SVC(),
            }

            params = {
                "Logistic Regression": {
                    'penalty': ['l2'],
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs'],
                },
                "Decision Tree Classifier": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                },
                "Random Forest Classifier": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                },
                "K-Nearest Neighbors Classifier": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                },
                "Bernoulli Naive Bayes": {
                    'alpha': [0.1, 0.5, 1.0],
                    'binarize': [0.0, 0.1, 0.5],
                },
                "Support Vector Classifier": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                },
            }

            model_report:dict= evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
            models=models,param=params)


            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(best_model_score,best_model_name)


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                
            )

            predicted=best_model.predict(X_test)

            acc = accuracy_score(y_test, predicted)
            return acc

        except Exception as e:
            raise CustomException(e,sys)








