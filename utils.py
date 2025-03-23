# *******************************************
# **** Assignment submission by Arjun Shrivatsan
# **** EAI 6010 - Assignment No: Module 5 - Face Mask Detection Microservice
# *******************************************

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model

# 🔁 Auto-download from Google Drive if not found locally
def maybe_download_model():
    model_path = "model/mask_detection.keras"
    if not os.path.exists(model_path):
        print("🔽 Model not found locally. Downloading from Google Drive...")
        import gdown
        os.makedirs("model", exist_ok=True)
        # Replace with your actual Google Drive file ID
        file_id = "1uAIfC2pCGBf8dAoytUrJ3IA0fY3dvH5D"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    else:
        print("✅ Model file already exists locally.")

# ✅ Load the Keras model (after making sure it's downloaded)
def load_model():
    maybe_download_model()
    print("✅ Loading real Keras model from model/mask_detection.keras")
    model = keras_load_model("model/mask_detection.keras")
    return model

# 🔍 Image preprocessing and prediction
def predict_mask(image, model):
    print("🔥 Real model is being used for prediction")

    # Convert to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    img_resized = cv2.resize(img_rgb, (224, 224))  # Adjust if your model needs a different size

    # Normalize and reshape
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    result = model.predict(img_array)[0]
    print(f"🧠 Raw model prediction output: {result}")

    # Binary or Softmax classifier
    if len(result) == 2:
        prediction = "Mask" if np.argmax(result) == 0 else "No Mask"
        confidence = float(np.max(result))
    else:
        prediction = "Mask" if result[0] > 0.5 else "No Mask"
        confidence = float(result[0] if prediction == "Mask" else 1 - result[0])

    print(f"✅ Prediction: {prediction}, Confidence: {confidence}")
    return prediction, confidence
