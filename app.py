import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("sickle_cell_model.h5")
    return model

model = load_model()

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (224, 224))  # Adjust based on model input size
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model input
    return image

# Streamlit UI
st.title("Sickle Cell Detection App")
st.write("Upload images of red blood cells to detect sickle cells.")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        class_label = "Sickle Cell Detected" if prediction[0][0] > 0.5 else "No Sickle Cells"
        
        # Display result
        st.write(f"## Prediction for {uploaded_file.name}: {class_label}")
        
        # If sickle cells are detected, show precautions and helpline
        if prediction[0][0] > 0.5:
            st.warning("⚠ Sickle Cell Detected! Consult a doctor immediately.")
            st.write("### Precautions:")
            st.write("- Stay hydrated and drink plenty of water.")
            st.write("- Avoid extreme temperatures.")
            st.write("- Follow a healthy diet and avoid stress.")
            st.write("- Regularly monitor your health with a doctor.")
            st.write("### Helpline Numbers:")
            st.write("📞 **India:** 108 (Emergency) | 104 (Health Helpline)")
            st.write("📞 **USA:** 1-800-CDC-INFO (1-800-232-4636)")
