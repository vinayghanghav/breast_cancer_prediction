import streamlit as st
import numpy as np
import pandas as pd
import pickle
import random

# --------------------------
# Load the trained model
# --------------------------
model_path = "best_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🎗️", layout="wide")

st.title("🎗️ Breast Cancer Prediction App")
st.markdown("""
This app predicts whether a breast tumor is **Benign** or **Malignant** 
based on features from a digitized image of a fine needle aspirate (FNA) of a breast mass.
""")

# --------------------------
# Feature names
# --------------------------
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

# --------------------------
# CSV Upload Section
# --------------------------
st.sidebar.header("📂 Upload CSV Data (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file containing feature data", type=["csv"])

default_values = []

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if all(col in data.columns for col in feature_names):
            random_row = data.sample(1, random_state=random.randint(0, 9999))
            default_values = random_row[feature_names].iloc[0].values
            st.sidebar.success("✅ CSV uploaded successfully. Random row loaded as default values.")
        else:
            st.sidebar.error("❌ CSV missing required feature columns. Random values will be used instead.")
            default_values = np.random.uniform(0.5, 15.0, size=len(feature_names))
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}. Random values will be used instead.")
        default_values = np.random.uniform(0.5, 15.0, size=len(feature_names))
else:
    # Generate random values if no CSV uploaded
    default_values = np.random.uniform(0.5, 15.0, size=len(feature_names))

# --------------------------
# Input fields for prediction
# --------------------------
st.header("🔢 Enter or adjust features for prediction")
cols = st.columns(3)
inputs = []

for i, feature in enumerate(feature_names):
    default_value = float(default_values[i])
    with cols[i % 3]:
        value = st.number_input(f"{feature.replace('_', ' ').title()}", value=default_value, format="%.5f", key=f"input_{i}")
        inputs.append(value)

# --------------------------
# Prediction
# --------------------------
if st.button("🔍 Predict"):
    try:
        input_data = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        # Show confidence if supported
        confidence = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0]
            confidence = np.max(prob) * 100

        result = "Malignant" if prediction == 1 else "Benign"

        st.subheader("🩺 Prediction Result")
        st.success(f"The tumor is predicted to be **{result}**.")

        if confidence:
            st.info(f"Model confidence: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
