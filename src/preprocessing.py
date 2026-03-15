import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = {}
    
    def handle_missing_val(self,df:pd.DataFrame) ->  pd.DataFrame:

        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object','category']).columns

        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())

        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        return df
    
    def encode_category(self,df:pd.DataFrame) -> pd.DataFrame:

        categorical_cols = df.select_dtypes(include=['object','category']).columns

        for col in categorical_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            self.label_encoder[col] = encoder
        
        return df
    
    def scale_features(self, x: pd.DataFrame) -> pd.DataFrame:

        numeric_cols = x.select_dtypes(include=['number']).columns
        x[numeric_cols] = self.scaler.fit_transform(x[numeric_cols])

        return X
    
    def split_data(self, df:pd.DataFrame,target_column:str):

        if target_column not in df.columns:
            raise ValueError("Target column not found in dataset")
        
        x = df.drop(columns=[target_column])
        y = df[target_column]

        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

        return x_train,x_test,y_train,y_test
    

        