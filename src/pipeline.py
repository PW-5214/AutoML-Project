import pandas as pd

from src.data_ingestion import DataIngestion
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.model_selection import ModelSelector
from src.training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.bias_detection import BiasDetector
from src.explainability import Explainability
from src.llm_engine import LLMEngine
from src.report_engine import ReportEngine
from src.eda import EDA
from src.hyperparameter_tuning import HyperparameterTuner
from src.model_leaderboard import ModelLeaderboard

class MLpipeline:

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.feature_engineer = FeatureEngineer()
        self.selector = ModelSelector()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.bias_detector = BiasDetector()
        self.eda = EDA()
        self.llm = LLMEngine()
        self.report_engine = ReportEngine()
        self.tuner = HyperparameterTuner()
        self.leaderboard = ModelLeaderboard()

        self.best_model = None
        self.best_model_name = None
    
    def run_pipeline(self,file_path,target_column,task="classification"):
        self.ingestion = DataIngestion(file_path)
        print("Step 1:Data Ingestion")
        df = self.ingestion.load_data()
        self.ingestion.validate_data(df)

        print("\nStep 2: Exploratory Data Analysis")
        self.eda.dataset_overview(df)
        self.eda.correlation_matrix(df)
        self.eda.distribution_plots(df)

        print("\nStep 3 : Preprocessing")
        df = self.preprocessor.handle_missing_val(df)
        df = self.preprocessor.encode_category(df)

        print("\nStep 4: Feature Engineering")
        x = df.drop(columns=[target_column])
        x = self.feature_engineer.generate_poly_features(x)
        x[target_column] = df[target_column]

        print("\nStep 5: Train Test Split")
        x_train,x_test,y_train,y_test = self.preprocessor.split_data(df,target_column)
        print(f"Training set size:{x_train.shape}")
        print(f"Test set size{x_test.shape}")

        print("\nStep 6: Feature Selection")
        x_train = self.feature_engineer.select_best_features(
            x_train,y_train,task=task.capitalize()
        )
        x_test = x_test[x_train.columns]

        print("\nStep 7: Model Selection & Training")
        self.selector = ModelSelector(task=task)
        results = self.selector.train_and_evaluate(x_train,x_test,y_train,y_test)
        print("\nModel Performance Results:")
        print(results)

        for idx,row in results.iterrows():
            model_name = row['Model']
            metrics = row.drop("Model").to_dict()
            self.leaderboard.add_model(model_name,metrics)
        self.leaderboard.display_leaderboard()


        print("\nStep 8 :Selecting Best Model")
        self.best_model_name = results.iloc[0]["Model"]
        self.best_model = self.selector.models[self.best_model_name]
        print(f"Best Model:{self.best_model_name}")

        print("\nStep 9: Hyperparameter Optimization")

        self.best_model = self.tuner.tune(
            self.best_model,
            self.best_model_name,
            x_train,
            y_train
            )

        print("\n Step 10: Training Best Model")
        self.best_model = self.trainer.train_model(self.best_model,x_train,y_train)

        print("\nStep 11: Saving model")
        model_path = self.trainer.save_model(self.best_model,self.best_model_name)

        print("\nStep 12:Model Evaluation")

        if task.lower() == "classification":
            evaluation_results = self.evaluator.evaluate_classification(self.best_model,x_test,y_test)
        else:
            evaluation_results = self.evaluator.evaluate_regression(self.best_model,x_test,y_test)
        
        print("\nEvaluation Results:")
        self.evaluator.print_results()

        print("\nStep 13: Bias Detection")

        bias_report= {}

        if target_column in df.columns:
            bias_report['class_distribution'] = (
                self.bias_detector.detect_class_imbalance(df,target_column).to_dict()
            )
        bias_report['feature_bias'] = self.bias_detector.detect_feature_bias(df)

        bias_report['numeric_skew'] = (
            self.bias_detector.detect_numeric_skew(df).to_dict()
        )

        print("Bias Detection Complete")

        print("\nStep 14: Model Explainability")

        explainability  = Explainability(self.best_model)
        feature_imp = explainability.feature_importance(x_test)

        if feature_imp is not None:
            print("\n Top Features:")
            print(feature_imp.head())
        
        print("\nStep 15: Generating Summary")
        
        llm_summary = self.llm.generate_final_summary(
            evaluation_results,
            bias_report,
            feature_imp
        )

        print("\n Step 16: Generating Report")

        report = self.report_engine.generate_text_report(
            evaluation_results,
            bias_report,
            feature_imp,
            llm_summary
        )

        report_path = self.report_engine.save_report(report)
        print("\n" + "=" *50)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("=" *50)
        print(f"Model saved at:{model_path}")
        print(f"Report saved at:{report_path}")

        return {
            "model":self.best_model,
            "model_name":self.best_model_name,
            "evaluation":evaluation_results,
            "bias_report":bias_report,
            "feature_importance":feature_imp,
            "report_path":report_path,
            "model_path":model_path,
        }
    
if __name__ == "__main__":

    pipeline = MLpipeline()

    results = pipeline.run_pipeline(
        file_path="data/raw/internet_speed.csv",
        target_column="Internet_speed",
        task="regression"
    )



        