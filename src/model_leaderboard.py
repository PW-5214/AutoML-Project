import pandas as pd

class ModelLeaderboard:

    def __init__(self):
        self.leaderboard_data = []

    def add_model(self,model_name:str,metrics:dict):
        entry = {
            "Model":model_name,
            **metrics
        }
        self.leaderboard_data.append(entry)
    
    def get_leaderboard(self,sort_by:str = None):
        df = pd.DataFrame(self.leaderboard_data)

        if sort_by and sort_by in df.columns:
            df = df.sort_values(by=sort_by,ascending=False).reset_index(drop=True)
        else:
            if 'R2_score' in df.columns:
                df = df.sort_values(by='R2_score',ascending=False).reset_index(drop=True)
            elif 'accuracy' in df.columns:
                df = df.sort_values(by='accuracy',ascending=False).reset_index(drop=True)
            
        df.insert(0,'Rank',range(1,len(df) + 1))
        return df
    
    def display_leaderboard(self):
        df = self.get_leaderboard()
        print("\n" + "=" *80)
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        return df
    
    def get_leaderboard_text(self):
        df = self.get_leaderboard()
        text = "\n" + "="*80 + "\n"
        text += "Model LeaderBoard - Performance Comparison\n"
        text += "="*80 + "\n"
        text += df.to_string(index=False) + '\n'
        text += "="*80 + '\n\n'
        return text
    
    def get_top_model(self):
        df = self.get_leaderboard()
        if len(df) > 0:
            top = df.iloc[0]
            return {
                "rank" : top['Rank'],
                "model" : top["Model"],
                "metrics":top.drop(["Rank","Model"]).to_dict()
            }
        return None
    
        

