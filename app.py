import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Define the custom loss function (if needed)
def custom_mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Load the model with the custom loss function
model_path = 'C:/Users/KIIT/Desktop/hydropowerdata/cnn_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path, custom_objects={'mse': custom_mse})
else:
    raise FileNotFoundError(f'Model file {model_path} not found.')

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form['data']
        data = np.array([float(x) for x in data.split(',')])
        data = data.reshape(1, -1, 1)  # Reshape for CNN model

        # Predict using the model
        prediction = model.predict(data)
        prediction = prediction.flatten()[0]  # Get the single value

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
