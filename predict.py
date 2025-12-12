
import pandas as pd
import joblib
import sys
import os
from model import LungCancerPredictor

def predict_new_patients(input_file, model_file='lung_cancer_model.joblib'):
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found. Please run model.py to train first.")
        return

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    print(f"Loading model from {model_file}...")
    # We need to verify how to load. The class has a load method but joblib.load returns the pipeline directly if we dumped the attribute.
    # In model.py: joblib.dump(self.model, filepath) -> dumps the Pipeline object.
    
    model = joblib.load(model_file)
    
    print(f"Loading input data from {input_file}...")
    new_data = pd.read_csv(input_file)
    
    # Preprocessing needs to happen. 
    # The pipeline handles scaling/encoding, but we need the manual steps (dropping cols, mapping stage).
    # We should instantiate the predictor class to use its helper methods, but we need to ensure the model validation matches.
    
    predictor = LungCancerPredictor()
    # We don't need to train, just use preprocess_data
    processed_data = predictor.preprocess_data(new_data)
    
    # Predict
    print("Predicting...")
    # The pipeline expects X (no target).
    if 'survived' in processed_data.columns:
        processed_data = processed_data.drop('survived', axis=1)
        
    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]
    
    # Save results
    results = new_data.copy()
    results['predicted_survival'] = predictions
    results['survival_probability'] = probabilities
    
    output_file = 'predictions_output.csv'
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    # Display first few
    print("\nSample Predictions:")
    print(results[['id', 'predicted_survival', 'survival_probability']].head())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_csv>")
        # For demo, if dataset_med.csv exists, we can start by predicting on a sample of it
        if os.path.exists('dataset_med.csv'):
            print("\nDemo: Predicting on first 5 rows of dataset_med.csv")
            sample = pd.read_csv('dataset_med.csv').head(5)
            sample.to_csv('sample_input.csv', index=False)
            predict_new_patients('sample_input.csv')
    else:
        predict_new_patients(sys.argv[1])
