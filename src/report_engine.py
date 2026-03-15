import os

class ReportEngine:

    def __init__(self,report_dir='reports'):
        self.report_dir = report_dir
        os.makedirs(self.report_dir,exist_ok=True)

    def generate_text_report(self,evaluation,bias,feature_importance,llm_summary):

        report = "=== Model Report ===\n\n"

        report += "Model Evaluation\n"
        report += "------------\n"

        for metric,value in evaluation.items():
            report += f"{metric}:{value}\n"
        
        report += "\nBias Detection\n"
        report += "-----------------\n"

        for key,value in bias.items():
            report += f"{key}:{value}\n"
        
        report += "\nFEATURE IMPORTANCE\n"
        report += "----------------\n"

        if feature_importance is not None:
            report += feature_importance.to_string(index=False)

        report += "\n\nLLM SUMMARY\n"
        report += "----------------\n"

        report += llm_summary

        return report
    
    def save_report(self,report_text,file_name="model_report.txt"):

        file_path = os.path.join(self.report_dir,file_name)

        with open(file_path,"w") as f:
            f.write(report_text)

        print(f"Report Saved at:{file_path}")
        return file_path
        
