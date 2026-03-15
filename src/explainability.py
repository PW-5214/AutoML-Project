import pandas as pd
import shap
import matplotlib.pyplot as plt

class Explainability:

    def __init__(self,model):
        self.model = model
        self.explainer = None
        self.shap_values = None
    
    def feature_importance(self,x:pd.DataFrame):

        if hasattr(self.model,"feature_importances_"):

            importance = self.model.feature_importances_

            feature_importance = pd.DataFrame({
                "feature":x.columns,
                "importance":importance
            }).sort_values(by='importance',ascending=False)
            return feature_importance
        else:
            print("Model does not support feature importance")
            return None
        
    def shap_explain(self,x:pd.DataFrame):
        self.explainer = shap.Explainer(self.model,x)
        self.shap_values = self.explainer(x)
        return self.shap_values

    def shap_summary_plots(self,x:pd.DataFrame):
        if self.shap_values is None:
            self.shap_explainS(x)
        shap.summary_plot(self.shap_values,x)
    
    def shape_feautures(self,feature_name:str,x:pd.DataFrame):

        if self.shap_values is None:
            self.shap_explain(x)
        shap.plots.scatter(self.shap_values[:,feature_name])
    