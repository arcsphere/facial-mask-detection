# *******************************************
# **** Assignment submission by Arjun Shrivatsan
# **** EAI 6010 - Assignment No: Module 5 - Face Mask Detection Microservice
# *******************************************

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model

# File path setup
MODEL_PATH = "model/mask_detection.keras"
GDRIVE_FILE_ID = "1uAIfC2pCGBf8dAoytUrJ3IA0fY3dvH5D"  # Replace with actual file ID

def download_model():
    """Download model from Google Drive using gdown if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("📦 Model file not found. Downloading from Google Drive...")
        try:
            import gdown
        except ImportError:
            print("📦 Installing gdown...")
            os.system("pip install gdown")
            import gdown

        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
            MODEL_PATH,
            quiet=False
        )
        print("✅ Model downloaded successfully!")

# Load the real Keras model
def load_model():
    download_model()
    print(f"✅ Loading real Keras model from {MODEL_PATH}")
    model = keras_load_model(MODEL_PATH)
    return model

# Image preprocessing and prediction
def predict_mask(image, model):
    print("🔥 Real model is being used for prediction")

    # Convert to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Normalize and reshape
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    result = model.predict(img_array)[0]
    print(f"🧠 Raw model prediction output: {result}")

    # Softmax for 3-class classification (mask, no mask, incorrect mask)
    prediction_labels = ["Mask", "No Mask", "Incorrect Mask"]
    prediction = prediction_labels[np.argmax(result)]
    confidence = float(np.max(result))

    print(f"✅ Prediction: {prediction}, Confidence: {confidence}")
    return prediction, confidence
