import os
import cv2
import requests
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model

def maybe_download_model():
    model_path = "model/mask_detection.h5"
    os.makedirs("model", exist_ok=True)

    if not os.path.exists(model_path):
        print("🔽 Model not found locally. Downloading from Dropbox...")

        # ✅ Dropbox direct download link (make sure ?dl=1)
        download_url = "https://www.dropbox.com/scl/fi/li1dvx1e16tp1ozwm7qld/mask_detection.h5?rlkey=zv4ajq4if8l2vp0h1oazy3pob&st=fjju9p5q&dl=1"

        response = requests.get(download_url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded successfully.")
        else:
            raise Exception(f"❌ Failed to download model. HTTP {response.status_code}")

def load_model():
    print("✅ Loading Keras model from model/mask_detection.h5")
    maybe_download_model()
    return keras_load_model("model/mask_detection.h5")

def predict_mask(image, model):
    print("🔥 Real model is being used for prediction")

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)[0]
    print(f"🧠 Raw model prediction output: {result}")

    if len(result) == 2:
        prediction = "Mask" if np.argmax(result) == 0 else "No Mask"
        confidence = float(np.max(result))
    else:
        prediction = "Mask" if result[0] > 0.5 else "No Mask"
        confidence = float(result[0] if prediction == "Mask" else 1 - result[0])

    print(f"✅ Prediction: {prediction}, Confidence: {confidence}")
    return prediction, confidence
