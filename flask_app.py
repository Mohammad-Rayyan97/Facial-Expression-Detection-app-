from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os

# ✅ Load the trained model
MODEL_PATH = "model/Facial_expression_detection.h5"  # Ensure this path is correct

if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: Model file is missing!")
    exit(1)  # Stop the app if model is missing

model = load_model(MODEL_PATH)

# ✅ Define emotion labels
classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

# ✅ Process frames and predict emotions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Read the image from the request
        image_file = request.files['image'].read()
        image = Image.open(io.BytesIO(image_file)).convert("L")  # Convert to grayscale
        image = image.resize((48, 48))
        image = np.asarray(image, dtype=np.float32) / 255.0  # Normalize to [0,1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.expand_dims(image, axis=-1) # Add channel dimension

        # ✅ Debugging: Print Image Shape
        print(f"🔍 Received Image Shape: {image.shape}")

        # ✅ Predict Emotion
        prediction = model.predict(image)
        print(f"🧠 Model Raw Output: {prediction}")

        predict_index = np.argmax(prediction)
        predict_class = classes[predict_index]
        confidence = round(float(prediction[0][predict_index]) * 100, 2)

        print(f"✅ Predicted Class: {predict_class}, Confidence: {confidence}%")

        return jsonify({"emotion": predict_class, "confidence": confidence})

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")  # Debugging print
        return jsonify({"error": str(e)})

# ✅ Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get port from Render
    app.run(host="0.0.0.0", port=port, debug=True)
