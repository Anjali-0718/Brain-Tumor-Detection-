from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.h5')
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", str(e))

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

tumor_info = {
    'glioma': {
        'description': 'Gliomas are tumors that arise from glial cells in the brain or spine. They are among the most common types of primary brain tumors.',
        'prevention': 'Avoid exposure to radiation, maintain a healthy diet, manage stress, and monitor for neurological symptoms.'
    },
    'meningioma': {
        'description': 'Meningiomas are typically benign tumors that develop from the meninges, the membranes surrounding the brain and spinal cord.',
        'prevention': 'Limit radiation exposure, maintain a healthy lifestyle, and seek regular checkups if at risk.'
    },
    'pituitary': {
        'description': 'Pituitary tumors develop in the pituitary gland and can affect hormone levels and bodily functions.',
        'prevention': 'While often not preventable, early diagnosis and hormonal monitoring can help in management.'
    },
    'notumor': {
        'description': 'No signs of tumor detected in the MRI scan.',
        'prevention': 'Continue regular health checkups and maintain a brain-healthy lifestyle.'
    }
}

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_tumor(image_path):
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
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('index.html', result='No file selected', confidence='', file_path='')

        file = request.files['file']
        if file:
            filename = file.filename
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_location)
            print("Saved file to:", file_location)

            result, confidence, description, prevention = predict_tumor(file_location)

            return render_template('index.html',
                                   result=result,
                                   confidence=f"{confidence*100:.2f}%",
                                   description=description,
                                   prevention=prevention,
                                   file_path=f'/static/uploads/{filename}')
    return render_template('index.html', result=None)
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
