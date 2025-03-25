# *******************************************
# **** Assignment submission by Arjun Shrivatsan
# **** EAI 6010 - Assignment No: Face Mask Detection - Deployment
# *******************************************

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies (for OpenCV and image processing)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY . .

# Ensure the model directory exists
RUN mkdir -p model

# Expose the default FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
