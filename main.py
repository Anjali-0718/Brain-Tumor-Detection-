from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
import gdown

app = Flask(_name_)
app.secret_key = 'your_secret_key_here'

# --- Model Loading ---
model_path = os.path.join(os.path.dirname(_file_), 'models', 'model.h5')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

file_id = '1p1_6z6hCCJRXsZSmu8KtCjMChztIjjao'
url = f'https://drive.google.com/uc?id={file_id}'

if not os.path.exists(model_path):
    print("Downloading model.h5 from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Load the model safely
model = None
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", str(e))

# --- Class Labels and Info ---
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

tumor_info = {
    'glioma': {
        'description': 'Gliomas arise from glial cells in the brain/spine. They are common brain tumors.',
        'prevention': 'Avoid radiation, manage diet and stress, monitor for symptoms.'
    },
    'meningioma': {
        'description': 'Meningiomas are usually benign tumors from the meninges.',
        'prevention': 'Limit radiation exposure, stay healthy, and checkups if at risk.'
    },
    'pituitary': {
        'description': 'Pituitary tumors grow in the pituitary gland, affecting hormones.',
        'prevention': 'Early diagnosis, hormonal checkups help manage them.'
    },
    'notumor': {
        'description': 'No signs of tumor in the scan.',
        'prevention': 'Maintain a healthy lifestyle and routine health checkups.'
    }
}

# --- Uploads Folder ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Prediction Logic ---
def predict_tumor(image_path):
    if model is None:
        raise RuntimeError("Model not loaded. Cannot perform prediction.")
    
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    predicted_label = class_labels[predicted_class_index]

    result = "No Tumor" if predicted_label == 'notumor' else f"Tumor: {predicted_label}"
    description = tumor_info[predicted_label]['description']
    prevention = tumor_info[predicted_label]['prevention']

    return result, confidence_score, description, prevention

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if model is None:
            return render_template('index.html', result='Model failed to load. Please try again later.', confidence='', file_path='')

        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('index.html', result='No file selected', confidence='', file_path='')

        file = request.files['file']
        if file:
            filename = file.filename
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_location)
            print("Saved file to:", file_location)

            try:
                result, confidence, description, prevention = predict_tumor(file_location)
                return render_template('index.html',
                                       result=result,
                                       confidence=f"{confidence*100:.2f}%",
                                       description=description,
                                       prevention=prevention,
                                       file_path=f'/static/uploads/{filename}')
            except Exception as e:
                return render_template('index.html', result=f'Prediction error: {str(e)}', confidence='', file_path='')

    return render_template('index.html', result=None)

# --- Run App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
