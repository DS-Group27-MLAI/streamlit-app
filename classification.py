import cv2
from glob import glob
import streamlit as st
import numpy as np
import requests
from PIL import Image
from utils import prepare_image_from_bytes
import config

anomaly_images_test = glob('images/anomaly/*')
normal_images_test = glob('images/normal/*')


def load_image(label, anomaly=True):
    if anomaly:
        image = cv2.imread(anomaly_images_test[label], cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(normal_images_test[label], cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    test_X = cv2.resize(gray, (224, 224))

    return test_X


def classification_inference(labels=[0, 1]):
    st.subheader("Using Classification")
    with st.expander("See Sample Images"):
        columns = st.columns(4)
        st.write("""
            The images above show sample anomaly images and normal images. 
        """)
        for ii, label in enumerate(labels):
            image = load_image(label)
            image_56x56 = cv2.resize(image, (56, 56))
            columns[ii].image(image_56x56, caption="Anomaly Image")

        for ii, label in enumerate(labels):
            image = load_image(label, anomaly=False)
            image_56x56 = cv2.resize(image, (56, 56))
            columns[ii + len(labels)].image(image_56x56, caption="Normal Image")

    st.subheader("Please upload the image")
    st.text("Send the model to the server and detect whether the image is \nnormal/anomaly from the best model")
    uploaded_file = st.file_uploader(label="Choose a file for Anomaly Detection", type='jpeg')
    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        # To read file as bytes:
        
        files = {"file": uploaded_file.getvalue()}
        res = requests.post(config.API_HOST + f"model/classification", files=files)
        is_anomaly = res.json()
        result = is_anomaly.get("output")
        col1.image(uploaded_file.getvalue(), caption="Input image Uploaded")
        viz_result = np.array(Image.open(is_anomaly.get("viz_output")))
        col2.image(viz_result, caption="Visualization indicating the image label")
        if result:
            col2.text("Anomaly Image")
        else:
            col2.text("Normal Image")
            
        inference_time = is_anomaly.get("inference_time")
            
        col2.markdown("**Model Name:** ResNet 5 Repeated Module (Existing DNN)")
        col2.markdown("**Model Inference Time:** " + str(inference_time))
        