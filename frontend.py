from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st

model = load_model("braintumourclassifier4th.h5")
classes = ['glioma','meningioma','notumor', 'pituitary']

st.title("Brain Tumor Classification App")

uploaded_file = st.file_uploader("Choose an image from your device", type="jpg")

if uploaded_file is not None:
    #Preprocess
    uploaded_image = Image.open(uploaded_file)

    if uploaded_image.mode != "RGB":
        uploaded_image = ImageOps.grayscale(uploaded_image).convert("RGB")

    uploaded_image = uploaded_image.resize((128, 128))

    img_array = img_to_array(uploaded_image)

    img_array = np.expand_dims(img_array, axis=0)

    #Normalize
    img_array /= 255.0

    #Prediction
    prediction = model.predict(img_array)
    predict_index = np.argmax(prediction)
    predicted_class = classes[predict_index]

    #Result
    st.image(uploaded_image,use_column_width=True)
    st.write(f"Prediction: {predicted_class}")
