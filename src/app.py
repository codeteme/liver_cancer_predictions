from flask import Flask, request, jsonify
import pickle
import pandas as pd
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# **Define Feature Groups (Same as in model_training.ipynb)**
ordinal_features = ['Alcohol_Consumption', 'Obesity', 'Healthcare_Access', 
                    'Preventive_Care', 'Seafood_Consumption']

binary_features = ['Smoking_Status', 'Hepatitis_B_Status', 'Hepatitis_C_Status',
                   'Diabetes', 'Screening_Availability', 'Treatment_Availability', 
                   'Liver_Transplant_Access', 'Herbal_Medicine_Use']

nominal_features = ['Country', 'Region', 'Gender', 'Rural_or_Urban', 'Ethnicity']

numerical_features = ['Population', 'Incidence_Rate', 'Mortality_Rate', 'Age', 'Cost_of_Treatment', 'Survival_Rate']

target_variable = ['Prediction']

# **Load Preprocessing Pipelines**
with open("ordinal_encoder.pkl", "rb") as f:
    ordinal_encoder = pickle.load(f)

with open("one_hot_encoder.pkl", "rb") as f:
    one_hot_encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# **Load Feature Names Used in Training**
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# **Load Models**
with open("rf_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

deep_learning_model = tf.keras.models.load_model("deep_learning_model.keras")

# **Preprocess Input Function**
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Apply Ordinal Encoding
    df[ordinal_features] = ordinal_encoder.transform(df[ordinal_features])

    # Apply Binary Encoding
    for col in binary_features:
        df[col] = df[col].map({
            'No': 0, 'Yes': 1,
            'Negative': 0, 'Positive': 1,
            'Non-Smoker': 0, 'Smoker': 1,
            'Not Available': 0, 'Available': 1,
        })

    # Apply One-Hot Encoding
    encoded_nominals = one_hot_encoder.transform(df[nominal_features])
    nominal_feature_names = one_hot_encoder.get_feature_names_out(nominal_features)
    df_nominal = pd.DataFrame(encoded_nominals, columns=nominal_feature_names, index=df.index)

    # Drop original categorical columns and join encoded ones
    df = df.drop(columns=nominal_features).join(df_nominal)

    # Apply Scaling to Numerical Features
    df[numerical_features] = scaler.transform(df[numerical_features])

    # **Ensure Correct Feature Order**
    df = df[feature_names]

    return df

# To Test If Flask is Running
@app.route("/")
def home():
    return jsonify({"message": "Flask API is Running!"})

# **API Endpoint for Random Forest Prediction**
@app.route('/predict/rf', methods=['POST'])
def predict_rf():
    data = request.json
    input_features = preprocess_input(data)
    prediction = rf_model.predict(input_features)
    return jsonify({"prediction": int(prediction[0])})

# **API Endpoint for Deep Learning Prediction**
@app.route('/predict/dl', methods=['POST'])
def predict_dl():
    data = request.json
    input_features = preprocess_input(data)
    probability = deep_learning_model.predict(input_features)[0, 0]
    prediction = 1 if probability >= 0.5 else 0
    return jsonify({"prediction": prediction, "probability": float(probability)})

# **Run Flask App**
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)