import os
import pandas as pd

class DataIngestion:
    def __init__(self,file_path:str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:

        if not os.path.exists(self.file_path):
            raise FileNotFoundError("Dataset not found at: ",self.file_path)

        file_extension = self.file_path.split(".")[-1]

        if file_extension == "csv":
            df = pd.read_csv(self.file_path)
        
        elif file_extension in ['xlsx','xls']:
            df = pd.read_excel(self.file_path)
        
        else:
            raise ValueError("unsupported format,Use csv or xlsx")
        
        return df
    
    def validate_data(self, df:pd.DataFrame) -> None:

        if df.empty:
            raise ValueError("Dataset is empty")
        
        if df.shape[1] < 2 :
            raise ValueError("Dataset must contain at least 2 columns")
        
        print("Data validation successful")
        print(f"Rows:{df.shape[0]},Columns:{df.shape[1]}")
    

