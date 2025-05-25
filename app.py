import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

import os
import streamlit as st

model_path = "model/model.keras"
st.write("Current working directory:", os.getcwd())
st.write("Model file exists:", os.path.exists(model_path))


model = load_model("model/model.keras")
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

st.title("🚦 Traffic Sign Recognition")
st.write("Upload a PNG image of a traffic sign (Turn Right, Stop, Yield, No Entry)")

uploaded_file = st.file_uploader("Choose image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"🧠 Predicted: **{predicted_class}** with {confidence:.2%} confidence.")
