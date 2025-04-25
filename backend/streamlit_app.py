import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load model
model = load_model('../Models/my_emotion_detection_model.h5')
labels = ['happy', 'neutral', 'sad', 'surprise']

st.title("Facial Emotion Detection")
st.write("Upload an image or use your webcam to detect emotions")

# Use webcam or file uploader
option = st.radio("Choose input source:", ["Upload Image", "Use Webcam"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale

elif option == "Use Webcam":
    webcam_image = st.camera_input("Take a photo")
    if webcam_image:
        image = Image.open(webcam_image).convert("L")  # Convert to grayscale

# Once we have an image
if image:
    st.image(image, caption="Input Image", use_container_width=True)
    img_resized = image.resize((48, 48))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(np.expand_dims(img_array, -1), 0)

    prediction = model.predict(img_array)
    emotion = labels[np.argmax(prediction)]

    st.success(f"Predicted Emotion: **{emotion.upper()}**")
    st.write("Confidence Scores:")
    for lbl, conf in zip(labels, prediction[0]):
        st.write(f"{lbl}: {conf:.2f}")
