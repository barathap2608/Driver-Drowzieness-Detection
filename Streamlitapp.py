import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Exact same mapping from your Colab code
inv_map = {0: "Closed", 1: "Open", 2: "no_yawn", 3: "yawn"}

# Exact same fatigue function from your Colab code
def classify_fatigue(pred_index):
    label = inv_map[pred_index]
    if label in ["Open", "no_yawn"]:
        return 0, "Alert"
    elif label == "yawn":
        return 1, "Mild Fatigue"
    elif label == "Closed":
        return 2, "Severe Fatigue"

# Load model
model = load_model(r"C:\Users\acer\OneDrive\python files\Final Project\best_mobilenet.h5")

st.title("Driver Drowsiness Detection")

uploaded_file = st.file_uploader("Upload a driver image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image")

    # Exact same preprocessing from your Colab code
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100

    stage, fatigue_label = classify_fatigue(predicted_index)

    st.write(f"**Predicted Class :** {inv_map[predicted_index]}")
    st.write(f"**Confidence      :** {confidence:.2f}%")
    st.write(f"**Fatigue Stage   :** {fatigue_label}")