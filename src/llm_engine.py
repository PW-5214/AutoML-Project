import json

class LLMEngine:

    def __init__(self):
        pass
    
    def summarize_evaluation(self,evaluation_result:dict):

        summary = "Model Evaluation Summary:\n"

        for metric,value in evaluation_result.items():
            summary += f"{metric}:{value}\n"
        return summary
    
        
    def summarize_bias(self,bias_report:dict):

        summary = "Bias Detection Summary:\n"

        for key,value in bias_report.items():
            summary += f"{key}:{value}\n"
        return summary
    
    def summarize_features_importance(self,feature_importance_df):
        if feature_importance_df is None:
            return "\nFeature importance could not be generated for this model\n."

        top_features = feature_importance_df.head(5)
        summary = "\nTop Important Features:\n"
        for _,row in top_features.iterrows():
            summary += f"{row['feature']} importance = {row['importance']}\n"
        
        return summary
    
    def generate_final_summary(self,evaluation,bias,feature_importance):
        report = ""

        report += self.summarize_evaluation(evaluation)
        report += self.summarize_bias(bias)
        report += self.summarize_features_importance(feature_importance)

        return report

        