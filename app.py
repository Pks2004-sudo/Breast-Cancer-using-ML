from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        features = [
            float(request.form['radius_mean']),
            float(request.form['texture_mean']),
            float(request.form['perimeter_mean']),
            float(request.form['area_mean']),
            float(request.form['smoothness_mean'])
        ]

        # Scale input
        final_features = scaler.transform([features])

        # Predict
        prediction = model.predict(final_features)

        result = "Malignant (Cancer Detected)" if prediction[0] == 1 else "Benign (No Cancer)"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text="Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)
