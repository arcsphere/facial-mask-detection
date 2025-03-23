# 🎓 EAI 6010 - Assignment Submission

**Course:** EAI 6010 - Applications of AI  
**Module:** 5 – Face Mask Detection Microservice  
**Student:** Arjun Shrivatsan (002028814)  
**Deployment:** [Live Demo on Render](https://facial-mask-detection.onrender.com)

---

## 📘 About This Project

This microservice detects whether a person is wearing a mask using deep learning. It accepts an image input and responds with a classification: `Mask`, `No Mask`, or `Incorrectly Worn Mask`, along with a confidence score.

It includes:
- Preprocessing of face region from images
- A trained InceptionV3-based model with softmax output
- Dockerized FastAPI application for API-based access

---

## 🎯 What This Assignment Demonstrates

- Image classification with transfer learning (InceptionV3)
- TensorFlow model training and export
- FastAPI microservice architecture
- Docker containerization
- RESTful API exposure with Swagger UI
- GitHub integration and model file management

---

## 🚀 Running Instructions

### ✅ Hosted on Render

**URL:** [`https://facial-mask-detection.onrender.com`](https://facial-mask-detection.onrender.com)

1. Open the link.
2. Navigate to `/docs` to access Swagger UI.
3. Use the `POST /predict/` endpoint to upload an image and get predictions.

---

### 💻 Run Locally (Docker)

```bash
# Clone the repo
git clone https://github.com/arcsphere/facial-mask-detection.git
cd facial-mask-detection

# Build Docker image
docker build -t face-mask-api .

# Run container
docker run -p 8000:8000 face-mask-api

# Open in browser
http://localhost:8000/docs
📡 Use via curl
bash
Copy
Edit
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -F 'file=@test_image.jpg' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data'
⚙️ Installation (Without Docker)
bash
Copy
Edit
# Clone the project
git clone https://github.com/arcsphere/facial-mask-detection.git
cd facial-mask-detection

# Create virtual environment
python3 -m venv maskenv
source maskenv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start the app
uvicorn app.app:app --reload
🧠 Tech Stack
Model: InceptionV3 (Transfer Learning)

Framework: FastAPI

Serving: Docker + Uvicorn

Image Processing: OpenCV, NumPy

Serialization: .keras model format

Deployment: Render (free tier)

Version Control: Git + GitHub

📁 Notes on Model File
Due to GitHub’s 100MB limit, the model file (mask_detection.keras) is excluded from version control.

To load model in production (Render or Docker):

python
Copy
Edit
import os
if not os.path.exists("model/mask_detection.keras"):
    import gdown
    gdown.download("https://drive.google.com/uc?id=YOUR_FILE_ID", "model/mask_detection.keras", quiet=False)
Replace YOUR_FILE_ID with the Google Drive file ID.

