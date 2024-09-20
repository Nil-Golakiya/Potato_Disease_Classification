from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
)

try:
  model = tf.saved_model.load("./models/1")
except OSError:
  try:
    from tensorflow.keras.models import load_model
    model = load_model("./models/1.keras")
  except OSError:
    raise RuntimeError("Model file not found or corrupt. Please ensure the model exists and is compatible.")

CLASS_NAMES = ["Early_Blight", "Late_Blight", "Healthy"]


def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File()):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    predictions = model.predict(image_batch)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(predictions[0]))  # Convert numpy.float32 to native Python float
    response_data = {"class": predicted_class, "confidence": confidence, "class_index": int(predicted_class_index)}
    return json.dumps(response_data)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)