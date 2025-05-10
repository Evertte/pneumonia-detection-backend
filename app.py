from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load model
model = load_model("mobilenet_model.h5")

IMG_SIZE = 224

# Set up app
app = FastAPI()

# CORS settings for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your React domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        label = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
        confidence = float(prediction[0][0])

        logging.info(f"Prediction made: {label} ({confidence:.2f})")

        return {
            "prediction": label,
            "confidence": confidence
        }
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return {"error": str(e)}