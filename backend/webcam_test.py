# Scripts to load the model and test it with webcam
import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('../Models/my_emotion_detection_model.h5')

# Define the labels for the emotions
labels = ['happy', 'neutral', 'sad', 'surprise']

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam
cap = cv2.VideoCapture(0)  # 0 for the default camera

# Check if the webcam is opened
if not cap.isOpened():
    print("❌ Webcam couldn't be opened. Try using 1 or 2 instead of 0.")
    exit()

print("✅ Webcam successfully opened. Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_normalized = roi_resized.astype('float32') / 255.0
        roi_reshaped = np.expand_dims(np.expand_dims(roi_normalized, -1), 0)

        prediction = model.predict(roi_reshaped)
        emotion = labels[np.argmax(prediction)]
        print("🧠 Predicted Emotion:", emotion)

        # Draw results on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Facial Emotion Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed successfully.")
# End of script