from flask import Flask, jsonify, render_template, request
import tensorflow as tf
import numpy as np

# Define a dictionary mapping integer labels to class names
label_map = {0: 'crack', 1: 'dot', 2: 'good', 3: 'joint'}

# Load the model
model = tf.keras.models.load_model('models/marble_mobilenet_5k.h5')

# Create the Flask app
app = Flask(__name__)

# Define the index route
@app.route('/')
def index():
    return render_template("index.html")

# Define the prediction route
@app.route('/prediction', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image from the request
        img = request.files['image'].read()
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.keras.applications.mobilenet.preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Make the prediction
        prediction = model.predict(img)
        predicted_label_idx = np.argmax(prediction, axis=-1)[0]

        class_names = ['crack', 'dot', 'good', 'joint']
        predicted_label = class_names[predicted_label_idx]

        # Return the prediction as a JSON response
        return render_template('prediction.html', label=predicted_label)


# Start the Flask app
if __name__ == '__main__':
    app.debug = True
    app.run()
