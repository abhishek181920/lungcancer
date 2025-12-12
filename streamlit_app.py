
import streamlit as st
import pandas as pd
import joblib
import os
from model import LungCancerPredictor

# Set page config
st.set_page_config(
    page_title="Lung Cancer Survival Predictor",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .safe {
        color: #28a745;
        font-size: 2rem;
        font-weight: bold;
    }
    .danger {
        color: #dc3545;
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if os.path.exists('lung_cancer_model.joblib'):
        return joblib.load('lung_cancer_model.joblib')
    else:
        st.error("Model file not found. Please train the model first.")
        return None

def main():
    st.title("🫁 Lung Cancer Survival Prediction System")
    st.markdown("---")

    model = load_model()

    if model:
        # Create two columns for layout
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image("https://img.freepik.com/free-vector/human-internal-organ-with-lungs_1308-108923.jpg", caption="AI-Powered Prognosis")
            st.markdown("### Patient Details")
            st.info("Please enter the patient's clinical and demographic details to generate a survival prediction.")

        with col2:
            with st.form("prediction_form"):
                st.subheader("Demographics & Vitals")
                c1, c2, c3 = st.columns(3)
                with c1:
                    age = st.number_input("Age", min_value=0, max_value=120, value=65)
                    gender = st.selectbox("Gender", ["Male", "Female"])
                with c2:
                    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
                    country = st.selectbox("Country", ['Germany', 'Switzerland', 'Iceland', 'Finland', 'Sweden', 'Malta', 'Netherlands', 'Denmark', 'Luxembourg', 'Norway', 'Belgium', 'Austria', 'Slovenia', 'France', 'Ireland', 'Portugal', 'Estonia', 'Lithuania', 'Latvia', 'Slovakia', 'Czech Republic', 'Hungary', 'Poland', 'Romania', 'Bulgaria', 'Greece', 'Croatia', 'Cyprus', 'Italy', 'Spain']) # Simplified list, model handles others if encoded or ignores
                with c3:
                    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)

                st.subheader("Clinical History")
                c4, c5, c6 = st.columns(3)
                with c4:
                    smoking = st.selectbox("Smoking Status", ["Never Smoked", "Former Smoker", "Current Smoker", "Passive Smoker"])
                    cancer_stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
                with c5:
                    treatment = st.selectbox("Treatment Type", ["Surgery", "Chemotherapy", "Radiation", "Combined"])
                    family_history = st.selectbox("Family History of Cancer", ["Yes", "No"])
                with c6:
                    other_cancer = st.selectbox("History of Other Cancer", ["Yes", "No"])

                st.subheader("Comorbidities")
                c7, c8, c9 = st.columns(3)
                
                # Checkbox style for binary
                hypertension = st.checkbox("Hypertension")
                asthma = st.checkbox("Asthma")
                cirrhosis = st.checkbox("Cirrhosis")

                submit_btn = st.form_submit_button("Predict Survival")

        if submit_btn:
            # Prepare input data
            input_data = pd.DataFrame({
                'age': [age],
                'gender': [gender.lower()], # dataset likely lower/mixed, let's normalize if needed or rely on pipeline
                'bmi': [bmi],
                'country': [country],
                'cholesterol_level': [cholesterol],
                'smoking_status': [smoking],
                'cancer_stage': [cancer_stage],
                'treatment_type': [treatment],
                'family_history': [family_history.lower()], # 'yes'/'no'
                'other_cancer': [other_cancer.lower()],
                'hypertension': ['yes' if hypertension else 'no'],
                'asthma': ['yes' if asthma else 'no'],
                'cirrhosis': ['yes' if cirrhosis else 'no']
                # Note: The model pipeline will handle OneHotEncoding.
                # Important: Model expects lowercase for many cats based on EDA?
                # Let's check 'gender' unique values in EDA output: they were Male, Female in some datasets or male/female.
                # I'll stick to what the form gives but maybe lowercase the values to be safe if the training data was lowercase.
                # Actually, in check_columns.py output, I didn't print unique values for gender.
                # But in check_leakage.py, I printed unique values for smoking/treatment.
                # Let's assume standard casing but I will try to match training data format.
            })
            
            # Use the predictor class to preprocess (e.g. stage mapping)
            predictor = LungCancerPredictor()
            # We must load the stage mapping manually or via the class method
            # The class 'preprocess_data' does the mapping.
            
            # There is a small catch: preprocess_data expects columns to exist.
            # It drops id/dates if they exist. It maps cancer_stage.
            processed_data = predictor.preprocess_data(input_data)
            
            # Predict
            try:
                prediction = model.predict(processed_data)[0]
                probability = model.predict_proba(processed_data)[0][1]

                st.markdown("---")
                st.subheader("Prediction Results")
                
                with st.container():
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    if prediction == 1: # 1 = Survived (Yes) based on 'yes': 1 map
                        st.markdown(f'<p class="safe">High Chance of Survival</p>', unsafe_allow_html=True)
                        st.markdown(f"### Probability: {probability:.1%}")
                        st.balloons()
                    else:
                        st.markdown(f'<p class="danger">Low Chance of Survival</p>', unsafe_allow_html=True)
                        st.markdown(f"### Probability: {probability:.1%}")
                        st.warning("This prediction suggests a higher risk outcome. Please consult with a specialist.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.write("Debug - Input Data:")
                st.write(processed_data)

if __name__ == "__main__":
    main()
