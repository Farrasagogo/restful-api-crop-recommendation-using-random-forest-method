import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib
import logging

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')  # Replace with your model path
label_encoder = joblib.load('label_encoder.pkl')  # Replace with your label encoder path

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return "Crop Recommendation API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        logging.debug(f"Received data: {data}")
        
        # Convert data to DataFrame
        input_data = pd.DataFrame(data)
        logging.debug(f"Input DataFrame: {input_data}")

        # Make predictions
        predictions = model.predict(input_data)
        logging.debug(f"Model predictions (encoded): {predictions}")

        # Decode predictions
        decoded_predictions = label_encoder.inverse_transform(predictions)
        logging.debug(f"Decoded predictions: {decoded_predictions}")

        # Create response
        response = {"predictions": decoded_predictions.tolist()}
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
