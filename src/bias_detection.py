import pandas as pd

class BiasDetector:
    def __init__(self):
        self.bias_report = {}

    def detect_class_imbalance(self,df:pd.DataFrame,target_column:str):
        distribution = df[target_column].value_counts(normalize=True)

        self.bias_report['class_distribution'] = distribution.to_dict()

        return distribution
    
    def detect_feature_bias(self, df:pd.DataFrame):
        categorcial_cols = df.select_dtypes(include=['object','category']).columns

        feature_bias ={}

        for col in categorcial_cols:
            distribution = df[col].value_counts(normalize=True)
            feature_bias[col] = distribution.to_dict()
        
        self.bias_report["feature_bias"] = feature_bias

        return feature_bias
    
    def detect_numeric_skew(self,df:pd.DataFrame):
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns
        skewness = df[numeric_cols].skew()
        self.bias_report['numeric_skew'] = skewness.to_dict()
        return skewness
    
    def get_report(self):
        return self.bias_report
