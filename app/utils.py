# *******************************************
# **** Assignment submission by Arjun Shrivatsan
# **** EAI 6010 - Assignment No: Module 5 - Face Mask Detection Microservice
# *******************************************

import cv2
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model

# Load the real Keras model from .h5
def load_model():
    print("✅ Loading real Keras model from model/mask_detection.keras")
    model = keras_load_model("model/mask_detection.keras")
    return model

# Image preprocessing and prediction
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
