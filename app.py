"""
Rising Waters: Flood Prediction System - Flask Web Application
================================================================
This Flask app loads the trained flood prediction model and scaler,
accepts weather/rainfall inputs from the user, and predicts whether
a flood is likely to occur.
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

# ---------------------------------------------------------------------------
# Flask app configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Base directory for resolving model file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load the trained model, scaler, and feature column names at startup
model = joblib.load(os.path.join(MODELS_DIR, "floods.save"))
scaler = joblib.load(os.path.join(MODELS_DIR, "transform.save"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "features.save"))


@app.route("/")
def home():
    """Render the landing page with project introduction."""
    return render_template("home.html")


@app.route("/predict")
def predict_page():
    """Render the prediction form page."""
    return render_template("index.html", features=feature_columns)


@app.route("/result", methods=["POST"])
def result():
    """
    Accept form data, scale features, run prediction, and redirect
    to the appropriate result page based on the prediction (0 or 1).
    """
    # Build a dictionary of feature values from the submitted form
    input_data = {}
    for feature in feature_columns:
        value = request.form.get(feature)
        if value is None or value.strip() == "":
            # Redirect back to form if any field is missing
            return redirect(url_for("predict_page"))
        input_data[feature] = float(value)

    # Create a DataFrame with the same column order used during training
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # Apply the saved scaler (transform only — scaler was fit during training)
    input_scaled = scaler.transform(input_df)

    # Make prediction using the saved model
    prediction = model.predict(input_scaled)[0]

    # Redirect to the appropriate result page
    if int(prediction) == 1:
        return redirect(url_for("chance"))
    return redirect(url_for("no_chance"))


@app.route("/chance")
def chance():
    """Render the flood risk detected warning page."""
    return render_template("chance.html")


@app.route("/no_chance")
def no_chance():
    """Render the no flood risk detected safe result page."""
    return render_template("no_chance.html")


if __name__ == "__main__":
    # Run the Flask development server
    app.run(debug=False, host="127.0.0.1", port=9000)
