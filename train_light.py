
import pandas as pd
from model import LungCancerPredictor
import joblib
import os

def train_light_model():
    print("Loading data...")
    df = pd.read_csv('dataset_med.csv')
    
    # Train extremely light model for deployment
    # Using n_estimators=10, max_depth=10 to keep file size small
    print("Training light model (n_estimators=10, max_depth=10)...")
    predictor = LungCancerPredictor()
    predictor.train(df, n_estimators=10, max_depth=10)
    
    print("Saving model...")
    predictor.save('lung_cancer_model.joblib')
    
    size_mb = os.path.getsize('lung_cancer_model.joblib') / (1024 * 1024)
    print(f"Model saved. Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    train_light_model()
