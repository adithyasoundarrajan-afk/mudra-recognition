import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
model = tf.keras.models.load_model("mudra_model.keras")
classes = ['Musti', 'Pataka', 'Sikhara', 'Simhamukha', 'Trisula']
st.title("Basic Mudra Predictor")
file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
if file:
    img = Image.open(file).convert('RGB')
    st.image(img, width=300)
    img_arr = tf.keras.utils.img_to_array(img.resize((128, 128)))
    img_arr = tf.expand_dims(img_arr, 0)
    prediction = model.predict(img_arr)
    predicted_mudra = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.success(f"Prediction: {predicted_mudra}")
    st.info(f"Confidence: {confidence:.1f}%")