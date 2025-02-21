import os
import requests

MODEL_PATH = "Facial_expression_detection.h5"
HUGGINGFACE_MODEL_URL = "https://huggingface.co/YOUR_USERNAME/YOUR_MODEL_REPO/resolve/main/Facial_expression_detection.h5"

# ✅ Check if the model exists, else download it from Hugging Face
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    response = requests.get(HUGGINGFACE_MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("✅ Model downloaded successfully!")

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# ✅ Load the model after downloading
model = load_model(MODEL_PATH)

classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image'].read()
        image = Image.open(io.BytesIO(image_file)).convert("L")
        image = image.resize((48, 48))
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)

        prediction = model.predict(image)
        predict_index = np.argmax(prediction)
        predict_class = classes[predict_index]
        confidence = round(float(prediction[0][predict_index]) * 100, 2)

        return jsonify({"emotion": predict_class, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
