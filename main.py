import uuid

import cv2
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image

import config
import inference


app = FastAPI()

model_list = [
    'models/vae/model_best_weights_anomaly_detection_vae_designed_completion.h5',
    'models/vae/model_best_weights_anomaly_detection_vae_existing.h5',
    'model_best_weights_anomaly_detection_convae_designed.h5',
    'model_best_weights_classification_resnet_existing_completion.h5'
]


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/{style}")
def get_image(style: str, file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    output, resized = inference.best_model(model_list, image)
    # name = f"/storage/{str(uuid.uuid4())}.jpg"
    # cv2.imwrite(name, output)
    return {"output": output}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)