import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("Facial_expression_detection.h5")

# Define class labels
classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Load OpenCV's Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit App Title
st.title("🎭 Real-Time Facial Expression Detection")

# Sidebar for Webcam Control
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

start_webcam = st.sidebar.button("Start Webcam")

if start_webcam:
    st.session_state.webcam_active = True  # Keep webcam running

# Stop Webcam Button
stop_webcam = st.sidebar.button("Stop Webcam")

if stop_webcam:
    st.session_state.webcam_active = False  # Stop webcam

# Video Stream
if st.session_state.webcam_active:
    cap = cv2.VideoCapture(0)

    # Placeholder for video
    frame_placeholder = st.empty()

    while cap.isOpened() and st.session_state.webcam_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Extract face ROI
            face = gray_frame[y:y+h, x:x+w]

            # Preprocess face
            face = Image.fromarray(face).resize((48, 48))
            face = np.asarray(face, dtype=np.float32) / 255.0  # Normalize
            face = np.expand_dims(face, axis=0)  # Add batch dimension
            face = np.expand_dims(face, axis=-1) # Add channel dimension

            # Predict Emotion
            prediction = model.predict(face)
            predict_index = np.argmax(prediction)
            predict_class = classes[predict_index]
            confidence = round(float(prediction[0][predict_index]) * 100, 2)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display predicted emotion
            label = f"{predict_class} ({confidence}%)"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Convert frame to RGB (Streamlit needs RGB format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the video frame in Streamlit
        frame_placeholder.image(frame, channels="RGB")

        # Allow Streamlit to update the UI properly
        if stop_webcam:
            break

    cap.release()
    cv2.destroyAllWindows()
    st.success("Webcam Stopped!")
