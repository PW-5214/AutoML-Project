import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

class ModelEvaluator:

    def __init__(self):
        self.results = {}
    
    def evaluate_classification(self,model,x_test:pd.DataFrame,y_test:pd.Series):
        y_pred = model.predict(x_test)

        accuracy= accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,average='weighted',zero_division=0)
        recall = recall_score(y_test,y_pred,average='weighted',zero_division=0)
        f1 = f1_score(y_test,y_pred,average='weighted',zero_division=0)
        cm = confusion_matrix(y_test,y_pred)

        try:
            y_prob = model.predict_proba(x_test)
            roc_auc = roc_auc_score(y_test,y_prob,multi_class='ovr')
        except:
            roc_auc=None

        self.results = {
            "accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "f1_score":f1,
            "roc_auc":roc_auc,
            "confusion_matrix":cm
        }

        return self.results
    
    def evaluate_regression(self,model,x_test:pd.DataFrame,y_test:pd.Series):

        y_pred = model.predict(x_test)

        mae =mean_absolute_error(y_test,y_pred)
        mse =mean_squared_error(y_test,y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test,y_pred)

        self.results = {
            "MAE":mae,
            "MSE":mse,
            "RMSE":rmse,
            "R2_score":r2
        }

        return self.results

    def print_results(self):
        for metric,value in self.results.items():
            print(f"{metric}:{value}")
