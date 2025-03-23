# *******************************************
# **** Assignment submission by Arjun Shrivatsan
# **** EAI 6010 - Assignment No: Module 5 - Face Mask Detection Microservice
# *******************************************

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_model, predict_mask

app = FastAPI(
    title="Face Mask Detection Microservice",
    description="Upload an image to detect if a person is wearing a face mask.",
    version="1.0.0"
)

# Allow CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load the model at startup
model = load_model()

@app.get("/")
def root():
    return {"message": "Welcome to the Face Mask Detection API!"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        prediction, confidence = predict_mask(frame, model)

        return JSONResponse(content={
            "prediction": prediction,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
