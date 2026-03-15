import pandas as pd
import json
from datetime import datetime
from typing import Dict, List

class Leaderboard:
    """
    Model Leaderboard System for tracking and comparing all trained models
    """
    
    def __init__(self):
        self.leaderboard = []
        self.timestamp = datetime.now()
    
    def add_model_results(self, model_name: str, score: float, task: str, 
                         evaluation_metrics: Dict, hyperparams: Dict = None):
        """
        Add a model's results to the leaderboard
        
        Args:
            model_name: Name of the model
            score: Primary performance score (accuracy/R2)
            task: Task type ('classification' or 'regression')
            evaluation_metrics: Full evaluation metrics dictionary
            hyperparams: Model hyperparameters
        """
        entry = {
            "Rank": None,  # Will be updated after sorting
            "Model": model_name,
            "Score": round(score, 6),
            "Task": task,
            "Timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Metrics": evaluation_metrics,
            "Hyperparameters": hyperparams or {}
        }
        self.leaderboard.append(entry)
    
    def get_leaderboard_df(self) -> pd.DataFrame:
        """
        Get leaderboard as a sorted DataFrame
        
        Returns:
            DataFrame with models ranked by score
        """
        df = pd.DataFrame(self.leaderboard)
        df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)
        
        # Reorder columns
        cols = ["Rank", "Model", "Score", "Task", "Timestamp"]
        return df[cols]
    
    def display_leaderboard(self, top_n: int = None):
        """
        Display the leaderboard in a formatted table
        
        Args:
            top_n: Show only top N models (None for all)
        """
        df = self.get_leaderboard_df()
        
        if top_n:
            df = df.head(top_n)
        
        print("\n" + "=" * 80)
        print("MODEL LEADERBOARD")
        print("=" * 80)
        print(df[["Rank", "Model", "Score", "Task"]].to_string(index=False))
        print("=" * 80)
    
    def get_top_model(self) -> Dict:
        """Get the best performing model"""
        if not self.leaderboard:
            return None
        
        df = self.get_leaderboard_df()
        top = df.iloc[0]
        return {
            "rank": top["Rank"],
            "name": top["Model"],
            "score": top["Score"]
        }
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get detailed model comparison"""
        df = self.get_leaderboard_df()
        return df[["Rank", "Model", "Score", "Metrics"]]
    
    def save_leaderboard(self, filepath: str):
        """
        Save leaderboard to JSON
        
        Args:
            filepath: Path to save leaderboard
        """
        df = self.get_leaderboard_df()
        
        # Convert DataFrame to serializable format
        data = {
            "timestamp": self.timestamp.isoformat(),
            "models": df.to_dict('records')
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        
        print(f"Leaderboard saved to {filepath}")
    
    def get_score_statistics(self) -> Dict:
        """Get statistics about model performance"""
        scores = [entry["Score"] for entry in self.leaderboard]
        
        return {
            "best_score": max(scores),
            "worst_score": min(scores),
            "average_score": sum(scores) / len(scores),
            "std_deviation": pd.Series(scores).std(),
            "total_models": len(scores)
        }
    
    def export_to_csv(self, filepath: str):
        """Export leaderboard to CSV"""
        df = self.get_leaderboard_df()[["Rank", "Model", "Score", "Task", "Timestamp"]]
        df.to_csv(filepath, index=False)
        print(f"Leaderboard exported to {filepath}")
