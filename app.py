import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the model (make sure to export your model from the notebook!)
MODEL_PATH = 'cnn_cifar10_model.h5'
model = load_model(MODEL_PATH)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(32, 32))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        confidence = predictions[0][class_index]

        os.remove(file_path)

        return jsonify({
            'class': class_names[class_index],
            'confidence': float(confidence)
        })


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)

# Let me know once you’re ready for the frontend code! 🚀
