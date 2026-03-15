import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest,f_classif,f_regression

class FeatureEngineer:

    def __init__(self,degree=2,k_features=50):
        self.degree = degree
        self.k_features = k_features
        self.poly = PolynomialFeatures(degree=self.degree,include_bias=False)


    def generate_poly_features(self,x:pd.DataFrame) -> pd.DataFrame:

        numeric_cols = x.select_dtypes(include=['number']).columns

        if len(numeric_cols) == 0:
            return x
        
        poly_features = self.poly.fit_transform(x[numeric_cols])
        features_names = self.poly.get_feature_names_out(numeric_cols)

        poly_df = pd.DataFrame(poly_features,columns=features_names)

        x = x.drop(columns=numeric_cols)
        x = pd.concat([x.reset_index(drop=True),poly_df.reset_index(drop=True)],axis=1)

        return x
    
    def select_best_features(self,x:pd.DataFrame,y,task="Classification"):

        k = min(self.k_features,x.shape[1])
        if task == "Classification":
            selector = SelectKBest(score_func=f_classif,k=k)
        else:
            selector = SelectKBest(score_func=f_regression,k=k)
        
        x_new = selector.fit_transform(x,y)
        selected_indices = selector.get_support(indices=True)
        selected_columns = x.columns[selected_indices]
        x_selected = pd.DataFrame(x_new,columns=selected_columns)
        return x_selected
    
    