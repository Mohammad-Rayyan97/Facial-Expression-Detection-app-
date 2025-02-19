from flask import Flask, Response, render_template
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

# Initialize Flask app
app = Flask(__name__)

# Function to capture frames and process them
def generate_frames():
    cap = cv2.VideoCapture(0)  # Access webcam

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert to grayscale
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
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get port from Render
    app.run(host="0.0.0.0", port=port, debug=True)

