from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint,uniform

class HyperparameterTuner:

    def __init__(self):
        self.param_grids = {
            "random_forest": {
                "n_estimators": randint(100,500),
                "max_depth": randint(5,30),
                "min_samples_split": randint(2,10),
                "min_samples_leaf": randint(1,5)
            },
            "xgboost": {
                "n_estimators": randint(100,400),
                "max_depth": randint(3,10),
                "learning_rate": uniform(0.01,0.3),
                "subsample": uniform(0.6,0.4)
            },
            "lightgbm": {
                "n_estimators": randint(100,400),
                "max_depth": randint(5,20),
                "learning_rate": uniform(0.01,0.3),
                "num_leaves": randint(20,150)
            }

        }

    def tune(self,model,model_name,x_train,y_train):

        if model_name not in self.param_grids:
            return model
            
        search = RandomizedSearchCV(
                model,
                self.param_grids[model_name],
                n_iter=20,
                cv=3,
                scoring='r2',
                n_jobs=-1,
                random_state=42
            )
        search.fit(x_train,y_train)

        return search.best_estimator_
    