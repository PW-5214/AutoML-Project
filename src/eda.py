import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:

    def __init__(self,output_dir = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir,exist_ok=True)
    
    def dataset_overview(self,df:pd.DataFrame):
        print("Dataset Shape",df.shape)
        print("\n Column Types:")
        print(df.dtypes)

        print("\nMissing Values:")
        print(df.isnull().sum())
        
        print("\nBasic Statistics:")
        print(df.describe(include='all'))

    def correlation_matrix(self,df:pd.DataFrame):
        numeric_df = df.select_dtypes(include=['number'])

        if numeric_df.shape[1] < 2:
            print("Not enough for the correlation")
            return
        plt.figure(figsize=(10,8))

        sns.heatmap(numeric_df.corr(),annot=False,cmap='coolwarm')
        save_path = os.path.join(self.output_dir,"correalation_matrix.png")

        plt.savefig(save_path)
        plt.close()

    
    def distribution_plots(self,df:pd.DataFrame):
        numeric_cols = df.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            plt.figure()
            sns.histplot(df[col],kde=True)
            save_path = os.path.join(self.output_dir,f"{col}_distribution.png")
            plt.savefig(save_path)
            plt.close()
        