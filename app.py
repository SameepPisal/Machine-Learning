import os
import base64
import numpy as np
from flask import Flask, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the trained model
model = load_model('models/marble_mobilenet_5k.h5')

# Define the path to the test data directory
test_dir = 'C:/Users/Administrator/Downloads/archive/dataset/test'

# Define the image size
img_size = (224, 224)

# Define the class names
class_names = ['crack', 'dot', 'good', 'joint']

# Define a route to handle the prediction request
@app.route('/')
# Define a route to handle the prediction request

def predict():
    # Load the test data and their labels
    x_test = []
    y_test = []
    image_paths = []
    image_types = []
    for class_dir in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_dir)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(class_path, filename)
                    img = load_img(file_path, target_size=img_size)
                    x_test.append(img_to_array(img))
                    y_test.append(class_dir)
                    image_paths.append(file_path)
                    image_types.append(filename.split('.')[-1])

    # Normalize the pixel values of the test images
    x_test_norm = np.array(x_test) / 255.0

    # Make predictions on the test images
    predicted_probs = model.predict(x_test_norm)
    predicted_labels = np.argmax(predicted_probs, axis=1)

    # Base64 encode the images
    encoded_images = []
    for image_path in image_paths:
        with open(image_path, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')
        encoded_images.append(encoded_image)

    # Render the prediction results in a template
    return render_template('prediction.html',
                           class_names=class_names,
                           predicted_labels=predicted_labels,
                           true_labels=y_test,
                           encoded_images=encoded_images,
                           image_types=image_types,
                           len=len
                           )



if __name__ == '__main__':
    app.run(debug=True)
