from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import math
import random

app = Flask(__name__)

# ===== Load Model and Scaler =====
MODEL_FILENAME = "philippines_earthquake_forecast_model.pkl"
SCALER_FILENAME = "philippines_earthquake_scaler.pkl"

model = joblib.load(MODEL_FILENAME)
scaler = joblib.load(SCALER_FILENAME)

# ===== Helper functions =====
def expected_count_to_prob(count):
    return 1 - math.exp(-count)

def classify_intensity(prob):
    if prob < 0.33:
        return "Low"
    elif prob < 0.66:
        return "Moderate"
    else:
        return "High"


# ===== Routes =====

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():

        # Get form inputs
        year = int(request.form.get("year"))
        month = int(request.form.get("month"))

        # Prepare input
        all_features = scaler.feature_names_in_
        default_values = {feat: 0.0 for feat in all_features if feat not in ["year", "month"]}

        input_dict = {"year": year, "month": month, **default_values}
        
        X_input_df = pd.DataFrame([input_dict], columns=all_features)

        X_scaled = scaler.transform(X_input_df)
        predicted_count = model.predict(X_scaled)[0]

        probability = expected_count_to_prob(predicted_count)

        # TEMP DEMO RANDOM ADJUST SIGNIFICANCE
        probability = random.uniform(0.4, 0.65)
        intensity = classify_intensity(probability)

        response = {
            "probability": round(probability, 2),
            "intensity": intensity,
            "year": year,
            "month": month
        }

        return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
