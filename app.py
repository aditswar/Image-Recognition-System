import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = None

def load_trained_model():
    global model
    inputs = tf.keras.Input(shape=(64, 64, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D(2, 2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.load_weights('model/weights.weights.h5')
    print("✅ Model loaded successfully.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    confidence = float(result[0][0])
    if confidence >= 0.5:
        return 'Dog', round(confidence * 100, 1)
    else:
        return 'Cat', round((1 - confidence) * 100, 1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type.'}), 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    label, confidence = predict_image(save_path)
    return jsonify({
        'prediction': label,
        'confidence': confidence,
        'image_url': f'/static/uploads/{filename}'
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    load_trained_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
