import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import cv2

st.set_page_config(page_title="Brain Tumor Prediction",page_icon="🧠",layout="wide")
st.header("Predicting Brain Tumor Type and Severity with Convolutional Neural Networks")

model = load_model('modelbrain.h5')

st.text("Please provide a brain MRI image for analysis.")
uploaded_file = st.file_uploader("Choose an Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image',width=540)
    st.write("Classifying...")

    st.write("")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL image to OpenCV format
    img = cv2.resize(img, (180, 180))  # Adjust the dimensions as per your model's input requirements

    # Normalize the image data
    img = img / 255.0  # Assuming your model expects input in the range [0, 1]

    # Expand dimensions to create a batch of one
    img = np.expand_dims(img, axis=0)

    # Perform the prediction
    prediction = model.predict(img)
    yclass= np.argmax(prediction, axis=1)

    if yclass == 0:
        st.write("The person has been diagnosed with a brain tumor, specifically of the type glioma.")
    elif yclass == 1:
        st.write("The person has been diagnosed with a brain tumor, specifically of the type meningioma")
    elif yclass == 2:
        st.write("No tumor has been detected.")
    else:
        st.write("The person has been diagnosed with a brain tumor, specifically of the type pituitary")
st.markdown("<p style='text-align: center; color: green; font-size: 14px; margin-top: 50px;'>DEVELOPED BY - BHAVYA SRI DUGGINA</p>", unsafe_allow_html=True)


