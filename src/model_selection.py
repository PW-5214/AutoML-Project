import pandas as pd

from sklearn.metrics import accuracy_score,r2_score

## Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

## Regeression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

##Boosting Models
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

class ModelSelector:

    def __init__(self,task = "classification"):
        self.task = task
        self.models = self._initialize_models()

    def _initialize_models(self):
        if self.task == "classification":

            models = {
                "logistic_regression":LogisticRegression(max_iter=1000),
                "random_forest":RandomForestClassifier(),
                "gradient_boosting":GradientBoostingClassifier(),
                "svm":SVC(),
                "knn":KNeighborsClassifier(),
                "naive_bayes":GaussianNB(),
                "decision_tree":DecisionTreeClassifier(),
                "xgboost":XGBClassifier(use_label_encoder=False,eval_metric='logloss'),
                "lightbgm":LGBMClassifier()
            }
        
        else:
            models ={
                "linear_regression":LinearRegression(),
                "random_forest":RandomForestRegressor(),
                "gradient_boosting":GradientBoostingRegressor(),
                "svr":SVR(),
                "knn":KNeighborsRegressor(),
                "decision_tree":DecisionTreeRegressor(),
                "xgboost":XGBRegressor(),
                "lightbgm":LGBMRegressor()
            }
        return models

    def train_and_evaluate(self,x_train,x_test,y_train,y_test):

        results = []

        for name,model in self.models.items():
            model.fit(x_train,y_train)
            predictions = model.predict(x_test)

            if self.task == "classification":
                score = accuracy_score(y_test,predictions)
            else:
                score = r2_score(y_test,predictions)

            results.append({
                "Model":name,
                "Score":score
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Score',ascending=False)

        return results_df
    
    def select_best_model(self,x_train,x_test,y_train,y_test):

        best_score = -float('inf')
        best_model = None
        best_model_name = None

        for name,model in self.models.items():
            model.fit(x_train,x_test,y_train,y_test)
            predictions = model.predict(x_test)

            if self.task == "classification":
                score = accuracy_score(y_test,predictions)
            else:
                score = r2_score(y_test,predictions)

            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name

        return best_model_name,best_model,best_score        
