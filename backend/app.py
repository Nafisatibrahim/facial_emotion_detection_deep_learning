import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
import os

# Load emotion detection model
model_path = os.path.join(os.path.dirname(__file__), '../Models/my_emotion_detection_model.h5')
model = load_model(model_path, compile=False)

labels = ['happy', 'neutral', 'sad', 'surprise']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Set page config
st.set_page_config(page_title="Facial Emotion Detection", page_icon="😄", layout="centered")

# Custom CSS for light background + dark text
st.markdown(
    """
    <style>
        /* Set full app background */
        .main {
            background-color: #0e1117;
        }

        /* Make all general text white */
        h1, h2, h3, h4, h5, h6, p, label, div, span {
            color: white !important;
        }

        /* Radio container label (e.g., Choose input method:) */
        .stRadio > label {
            color: white !important;
            font-weight: 500;
        }

        /* Radio options container */
        .stRadio > div {
            background-color: #1e1e1e !important;
            padding: 10px;
            border-radius: 10px;
        }

        /* Radio button labels */
        .stRadio > div label {
            color: white !important;
        }

        /* Style the file uploader label */
        section[data-testid="stFileUploader"] label {
            color: white !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #1e90ff !important;
            color: white !important;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Sidebar with options
st.sidebar.title("Settings")
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
show_crop = st.sidebar.checkbox("Show Cropped Face", value=False)

# Title
st.title("😄 Facial Emotion Detection")
st.write("Upload an image or use your webcam to detect emotions using a deep learning model.")

# Input source
input_type = st.radio("Choose input method:", ["Upload Image", "Take a Snapshot"])
image = None

# Snapshot feature with optional retake
if input_type == "Take a Snapshot":
    photo = st.camera_input("Take a picture")
    if photo:
        if st.button("Retake"):
            st.rerun()
        image = Image.open(photo)
elif input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

# If we have an image
if image:
    # Convert to OpenCV format
    img_cv = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected. Try again.")
    else:
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(np.expand_dims(face_normalized, -1), 0)

            prediction = model.predict(face_input)
            emotion = labels[np.argmax(prediction)]

            st.success(f"**Detected Emotion: {emotion.upper()}**")

            if show_confidence:
                st.subheader("Confidence Scores:")
                for label, score in zip(labels, prediction[0]):
                    st.write(f"{label}: {score:.2f}")

            if show_crop:
                st.image(face_resized, caption="Cropped Face (48x48)", width=100, clamp=True)

            # Draw bounding box + label
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_cv, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        result_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        st.image(result_img, caption="Result", use_container_width=True)
