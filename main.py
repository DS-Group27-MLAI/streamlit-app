"""
Adapted from Source: 
https://testdriven.io/blog/fastapi-streamlit/

"""

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

# in order of performance
anomaly_detection_models = [
    'models/vae/model_best_weights_anomaly_detection_vae_designed.h5',
    'models/vae/model_best_weights_anomaly_detection_vae.h5',
    'models/convae/model_best_weights_anomaly_convae.h5'
]

# in order of performance
classification_models = [
    'models/resnet/model_best_weights_classification_resnet_existing_completion.h5',
    'models/densenet/model_best_weights_classification_densenet_existing_completion.h5'
]

best_model_anomaly_detection = 1
best_model_classification = 0

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/model/{style}")
def get_image(style: str, file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    if style == "anomaly_detection":
        image = cv2.resize(image, (224,224))
        output, viz_output, image_label = inference.ad_best_model(best_model_anomaly_detection, anomaly_detection_models, image)
    elif style == "classification":
        image = cv2.resize(image, (224,224))
        output, viz_output, image_label = inference.classification_best_model(best_model_classification, classification_models, image)
    
    return {"output": output, "viz_output": viz_output, "label": image_label}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)