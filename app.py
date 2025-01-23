from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model("mnist_digit_model.h5")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Convert the image to grayscale and resize it to 28x28
    image = Image.open(io.BytesIO(file.read())).convert('L')
    image = image.resize((28, 28))  # MNIST model expects 28x28 input
    image = np.array(image) / 255.0  # Normalize the image
    image = image.reshape(1, 28, 28, 1)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    return jsonify({'digit': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)
