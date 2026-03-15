import os
import joblib

class ModelTrainer:

    def __init__(self,model_dir ="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir,exist_ok=True)
    
    def train_model(self,model,x_train,y_train):
        model.fit(x_train,y_train)
        return model

    def save_model(self,model,model_name):
        model_path = os.path.join(self.model_dir,f"{model_name}.pkl")
        joblib.dump(model,model_path)
        print(f"Model saved at:{model_path}")

        return model_path
    
    def load_model(self,model_path):
        model = joblib.load(model_path)
        return model

        