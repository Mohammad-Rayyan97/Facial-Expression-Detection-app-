import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("Facial_expression_detection.h5")

# Define class labels
classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit App Title
st.title("🎭 Facial Expression Detection")

# 📷 Take a Picture Using Streamlit Camera Input
uploaded_image = st.camera_input("Take a picture to detect emotion")

if uploaded_image is not None:
    # Convert the uploaded image to OpenCV format
    image = Image.open(uploaded_image).convert("RGB")
    image = np.array(image)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected. Please try again.")
    else:
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

            # Display predicted emotion
            st.success(f"🎭 Predicted Emotion: **{predict_class}** ({confidence}%)")

            # Draw rectangle around the face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show image with bounding box
            st.image(image, caption="Detected Face", use_column_width=True)
