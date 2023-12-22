from flask import Flask, request, jsonify
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('dentalify-gigimulut-xception256-13.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return "Dentalify API is running!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "Send a POST request with an image to get predictions."

    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['file']

        # Save the image to a temporary file
        file_path = 'temp_image.jpg'
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make predictions using the loaded model
        predictions = model.predict(img_array)

        # Get the class with the highest probability
        predicted_class = np.argmax(predictions)

        # Return the result as JSON
        result = {'class': str(predicted_class), 'probabilities': predictions.tolist()}
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

