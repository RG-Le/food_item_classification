import tensorflow as tf
import numpy as np
import uvicorn
import pickle

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Loading the model
def load_model(path):
    model = tf.keras.models.load_model(path)
    return model


# Loading Class Names
def load_classes(file):
    with open(file, "rb") as f:
        class_names = pickle.load(f)
    return class_names


# Initializing the model and classnames
model = load_model("First_Model.h5")
class_names = load_classes("class_names")


def read_process_image(data):
    img = Image.open(BytesIO(data))
    image = img.resize((224, 224))
    if image.mode == "RGBA":
        image = image.convert("RGB")
    return np.array(image)


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    img = read_process_image(await file.read())
    pred_prob = model.predict(tf.expand_dims(img, axis=0))
    pred_class = class_names[tf.argmax(pred_prob[0])]
    confidence = pred_prob[0][np.argmax(pred_prob[0])]
    return {
        'class': pred_class,
        'confidence': float(confidence)
    }


@app.get("/ping")
def ping():
    print("Hello World!!!!!!")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)