from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model/model.h5')

# Path to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class names (same as training folder names)
class_names = ['animals', 'birds', 'electronics', 'trees', 'vehicle']

# Confidence threshold for filtering irrelevant images
CONFIDENCE_THRESHOLD = 0.2

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array)[0]
            confidence = np.max(pred)
            predicted_class = class_names[np.argmax(pred)]

            # Check confidence
            if confidence < CONFIDENCE_THRESHOLD:
                prediction = 'No class'
            else:
                prediction = predicted_class

            return render_template('result.html', prediction=prediction, image_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
