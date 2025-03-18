import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)
model = load_model('braintumourclassifier5th.h5')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            img = preprocess_image(file_path)
            prediction = model.predict(img)
            class_idx = np.argmax(prediction, axis=1)[0]
            classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
            result = classes[class_idx]
            return render_template('result.html', result=result, image_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
