from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('crop_recommendation_model.pkl')
scaler = joblib.load('scaler.pkl')          # VERY IMPORTANT
le = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [
        float(request.form['nitrogen']),
        float(request.form['phosphorus']),
        float(request.form['potassium']),
        float(request.form['temperature']),
        float(request.form['humidity']),
        float(request.form['ph']),
        float(request.form['rainfall'])
    ]
    print("Input Data:", input_data)  # <-- check if values are received correctly

    input_array = np.array([input_data])
    print("Input Array Shape:", input_array.shape)

    input_scaled = scaler.transform(input_array)
    print("Scaled Input:", input_scaled)

    prediction = model.predict(input_scaled)
    crop_name = le.inverse_transform(prediction)[0]
    print("Predicted Crop:", crop_name)

    return render_template('index.html', prediction_text=f"Recommended Crop: {crop_name}")

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
    
