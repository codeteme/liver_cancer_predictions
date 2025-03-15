import streamlit as st
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

# Load encoders and model
with open("ordinal_encoder.pkl", "rb") as f:
    ordinal_encoder = pickle.load(f)

with open("one_hot_encoder.pkl", "rb") as f:
    one_hot_encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("rf_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

deep_learning_model = tf.keras.models.load_model("deep_learning_model.keras")

# Define feature categories
ordinal_features = ['Alcohol_Consumption', 'Obesity', 'Healthcare_Access', 'Preventive_Care', 'Seafood_Consumption']
binary_features = ['Smoking_Status', 'Hepatitis_B_Status', 'Hepatitis_C_Status', 'Diabetes', 'Screening_Availability', 'Treatment_Availability', 'Liver_Transplant_Access', 'Herbal_Medicine_Use']
nominal_features = ['Country', 'Region', 'Gender', 'Rural_or_Urban', 'Ethnicity']
numerical_features = ['Population', 'Incidence_Rate', 'Mortality_Rate', 'Age', 'Cost_of_Treatment', 'Survival_Rate']

# Define choices for categorical variables
ordinal_options = {
    'Alcohol_Consumption': ['Low', 'Moderate', 'High'],
    'Obesity': ['Underweight', 'Normal', 'Overweight', 'Obese'],
    'Healthcare_Access': ['Poor', 'Moderate', 'Good'],
    'Preventive_Care': ['Poor', 'Moderate', 'Good'],
    'Seafood_Consumption': ['Low', 'Medium', 'High']
}

binary_options = {
    'Smoking_Status': ['Non-Smoker', 'Smoker'],
    'Hepatitis_B_Status': ['Negative', 'Positive'],
    'Hepatitis_C_Status': ['Negative', 'Positive'],
    'Diabetes': ['No', 'Yes'],
    'Screening_Availability': ['Not Available', 'Available'],
    'Treatment_Availability': ['Not Available', 'Available'],
    'Liver_Transplant_Access': ['Not Available', 'Available'],
    'Herbal_Medicine_Use': ['No', 'Yes']
}

nominal_options = {
    'Country': ['United States', 'India', 'Brazil', 'France', 'China', 'Germany', 'Japan', 'Nigeria', 'Ethiopia', 'Mexico'],
    'Region': ['North America', 'Europe', 'Asia', 'Africa', 'South America'],
    'Gender': ['Male', 'Female'],
    'Rural_or_Urban': ['Rural', 'Urban'],
    'Ethnicity': ['Caucasian', 'African', 'Asian', 'Hispanic', 'Mixed']
}

# Streamlit App
st.title("Liver Cancer Prediction")

# Input fields for user
st.subheader("🔹 Enter Patient Details")

# Ordinal Feature Inputs
user_data = {}
for feature in ordinal_features:
    user_data[feature] = st.selectbox(f"{feature}:", ordinal_options[feature])

# Binary Feature Inputs
for feature in binary_features:
    user_data[feature] = st.selectbox(f"{feature}:", binary_options[feature])

# Nominal Feature Inputs
for feature in nominal_features:
    user_data[feature] = st.selectbox(f"{feature}:", nominal_options[feature])

# Numerical Feature Inputs
for feature in numerical_features:
    user_data[feature] = st.number_input(f"{feature}:", min_value=0.0, format="%.2f")

# Preprocess Input
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Apply Ordinal Encoding
    df[ordinal_features] = ordinal_encoder.transform(df[ordinal_features])

    # Apply Binary Encoding
    for col in binary_features:
        df[col] = df[col].map({'No': 0, 'Yes': 1, 'Negative': 0, 'Positive': 1, 'Non-Smoker': 0, 'Smoker': 1, 'Not Available': 0, 'Available': 1})

    # Apply One-Hot Encoding
    encoded_nominals = one_hot_encoder.transform(df[nominal_features])
    nominal_feature_names = one_hot_encoder.get_feature_names_out(nominal_features)
    df_nominal = pd.DataFrame(encoded_nominals, columns=nominal_feature_names, index=df.index)

    # Drop original categorical columns and join encoded ones
    df = df.drop(columns=nominal_features).join(df_nominal)

    # Apply Scaling to Numerical Features
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Ensure column order matches training set
    df = df[feature_names]

    return df

# Button to make predictions
if st.button("Predict"):
    input_features = preprocess_input(user_data)

    # Deep Learning Prediction
    dl_pred = deep_learning_model.predict(input_features)
    dl_result = f"Score: {dl_pred[0, 0]:.2f}"

    # Display Results
    st.subheader("Prediction Results")
    st.write(f"Deep Learning:{dl_result}")