# ***********************************************
# 🎓 Assignment Submission by Arjun Shrivatsan (002028814)
# 📚 Course: EAI 6010 - Applications of AI
# 📦 Module 5: Face Mask Detection - Microservice Deployment
# ***********************************************

## 🧠 About the Project

This microservice detects whether a person in an image is:
- Wearing a mask correctly 😷  
- Not wearing a mask ❌  
- Wearing a mask incorrectly ⚠️  

It utilizes a deep learning model trained using **InceptionV3** on a labeled dataset of masked and unmasked faces, deployed via **FastAPI** and **Docker**.

---

## ✅ What Does This Assignment Do?

- 📦 **Trains a Keras model** using OpenCV-preprocessed face regions.
- ⚙️ **Converts the model** to `.keras` format for compatibility.
- 🚀 **Deploys the model as a FastAPI microservice** using Docker.
- 🌐 **Enables predictions through a Swagger UI** and HTTP endpoints.

---

## 🚀 Running Instructions

### 🔁 A. Running on Render (Swagger UI)

> Deployed version (hosted):  
📍 `https://<your-render-url>.onrender.com/docs`  

- Upload an image via `/predict` endpoint.
- Receive `prediction` and `confidence` score.

---

### 💻 B. Running Locally

1. **Build the Docker image:**
   ```bash
   docker build -t face-mask-api .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 face-mask-api
   ```

3. **Access Swagger UI:**
   [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 🧪 C. cURL Test from Terminal

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -F 'image=@/path/to/your/image.jpg' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data'
```

---

## 📦 Installation (For Local Dev - Optional)

```bash
python -m venv maskenv
source maskenv/bin/activate   # or maskenv\Scripts\activate on Windows

pip install -r requirements.txt
uvicorn app.app:app --reload
```

---

## 🛠️ Tech Stack

| Layer             | Tech Used              |
|------------------|------------------------|
| Backend API      | FastAPI + Uvicorn      |
| ML Framework     | TensorFlow + Keras     |
| Image Processing | OpenCV                 |
| Deployment       | Docker, Render         |
| Swagger UI       | Built-in with FastAPI  |
| Model Architecture | InceptionV3 (Transfer Learning) |

---

## 📚 Resources and Citations

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Model Saving/Loading](https://keras.io/guides/serialization_and_saving/)
- [Face Mask Detection Dataset (MAFA-style annotations)](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [OpenCV Docs](https://docs.opencv.org/)

---

🧠 *Built with responsibility to deploy machine learning as a real-world accessible service.*
