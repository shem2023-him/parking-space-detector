import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("parking_space_model.h5")
class_labels = ['empty', 'occupied']

# Streamlit UI
st.title("🚗 Parking Space Detector")
st.write("Upload a parking lot image to check whether it's empty or occupied.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img_resized = image.resize((150, 150))  # Match training size
    img_array = np.array(img_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 150, 150, 3)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[int(prediction[0] > 0.5)]
    confidence = float(np.max(prediction)) * 100

    # Output
    st.subheader(f"Prediction: **{predicted_class.upper()}**")
    st.write(f"Confidence: `{confidence:.2f}%`")
