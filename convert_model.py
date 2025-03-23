import os
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model

print("📦 TensorFlow version:", tf.__version__)

# Paths
original_model_path = "model/mask_detection.h5"
converted_model_path = "model/mask_detection.keras"

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

print("🔄 Loading .h5 model from:", original_model_path)
model = load_model(original_model_path, compile=False)

print("💾 Saving compatible .keras model to:", converted_model_path)
save_model(model, converted_model_path)

print("✅ Model successfully saved.")
