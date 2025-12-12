
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

class LungCancerPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.stage_mapping = {
            'Stage I': 1,
            'Stage II': 2,
            'Stage III': 3,
            'Stage IV': 4
        }

    def preprocess_data(self, df):
        # Create a copy
        data = df.copy()

        # Drop ID and dates
        drop_cols = ['id', 'diagnosis_date', 'end_treatment_date']
        data = data.drop(columns=[c for c in drop_cols if c in data.columns])

        # Map Cancer Stage to ordinal
        if 'cancer_stage' in data.columns:
            # Handle potential string variations if needed, but assuming standard format
            data['cancer_stage'] = data['cancer_stage'].map(self.stage_mapping)
            # Fill unmapped (if any) with mode or median? Or treat as missing.
            # For now assume data is clean as seen in EDA (0 missing)
            
        return data

    def build_pipeline(self, X, n_estimators=100, max_depth=20):
        # Identify column types
        numeric_features = ['age', 'bmi', 'cholesterol_level', 'cancer_stage'] # cancer_stage is now numeric
        categorical_features = ['gender', 'country', 'smoking_status', 'treatment_type']
        
        # Binary yes/no features - handled as categorical or if already mapped? 
        # The dataset seems to have 'yes'/'no' or similar. 
        # I'll treat them as categorical for OneHot to be safe, or detect 2 unique values.
        # Let's list them explicitly based on description
        binary_features = ['family_history', 'hypertension', 'asthma', 'cirrhosis', 'other_cancer']
        
        # Merge binary into categorical for OneHotEncoder (it handles them fine)
        categorical_features.extend(binary_features)

        # Remove columns not in X
        numeric_features = [c for c in numeric_features if c in X.columns]
        categorical_features = [c for c in categorical_features if c in X.columns]

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Random Forest Classifier
        # Using n_jobs=-1 for parallel processing
        # Limiting depth or n_estimators for speed on large dataset (890k rows)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)

        model = Pipeline(steps=[('preprocessor', self.preprocessor),
                                ('classifier', clf)])
        return model

    def train(self, df, n_estimators=100, max_depth=20):
        # Preprocess manually for stage mapping
        data = self.preprocess_data(df)
        
        X = data.drop('survived', axis=1)
        y = data['survived']

        # Ensure y is proper format (0/1) if it's strings
        if y.dtype == 'object':
            y = y.map({'yes': 1, 'no': 0})

        # Split
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build pipeline
        self.model = self.build_pipeline(X_train, n_estimators=n_estimators, max_depth=max_depth)

        # Train
        print("Training model (this may take a while)...")
        self.model.fit(X_train, y_train)

        # Evaluate
        print("Evaluating...")
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy

    def save(self, filepath='lung_cancer_model.joblib'):
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath='lung_cancer_model.joblib'):
        self.model = joblib.load(filepath)

if __name__ == "__main__":
    # Test run
    df = pd.read_csv('dataset_med.csv')
    predictor = LungCancerPredictor()
    predictor.train(df)
    predictor.save()
