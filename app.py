from flask import Flask, request, render_template, jsonify
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load your trained model (update path as needed)
MODEL_PATH = 'model/cnn_cat_dog.h5'
model = None

def load_trained_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    else:
        print(f"⚠️  Model not found at '{MODEL_PATH}'. Train & save your model first.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0                        # rescale like training
    img_array = np.expand_dims(img_array, axis=0)        # add batch dimension
    result = model.predict(img_array)
    confidence = float(result[0][0])
    if confidence >= 0.5:
        label = "Dog"
        prob = round(confidence * 100, 1)
    else:
        label = "Cat"
        prob = round((1 - confidence) * 100, 1)
    return label, prob

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train and save your model first.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a JPG or PNG image.'}), 400

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
    os.makedirs('model', exist_ok=True)
    load_trained_model()
    app.run(debug=True)
