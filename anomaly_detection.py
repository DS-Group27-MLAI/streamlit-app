from keras.models import load_model
import cv2
from glob import glob
import streamlit as st
import numpy as np
from utils import prepare_image_from_bytes
import requests
from PIL import Image
import config

anomaly_images_test = glob('images/anomaly/*')
normal_images_test = glob('images/normal/*')


def load_image(label, anomaly=True):
    if anomaly:
        image = cv2.imread(anomaly_images_test[label], cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(normal_images_test[label], cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    test_X = cv2.resize(gray, (56, 56))

    return test_X


def anomaly_detection_inference(labels=[0, 1]):
    st.subheader("As an Anomaly Detection")
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
            columns[ii+len(labels)].image(image_56x56, caption="Normal Image")

    st.subheader("Please upload the image")
    st.text("Send the model to the server and detect whether the image is \nnormal/anomaly from the best model")
    st.text("Please upload the images to get another image output, so that SSIM Loss can be \nevaluated")
    uploaded_file = st.file_uploader(label="Choose a file for Anomaly Detection", type='jpeg')
    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        # To read file as bytes:
        
        files = {"file": uploaded_file.getvalue()}
        res = requests.post(config.API_HOST + f"model/anomaly_detection", files=files)
        img_path = res.json()
        result = np.array(Image.open(img_path.get("output")))
        col1.image(result, caption='Output Image from network')
        viz_result = np.array(Image.open(img_path.get("viz_output")))
        col2.image(viz_result, caption='Visualization Output Image from network')
        col2.text(img_path.get("label") + " Image")
        
        inference_time = img_path.get("inference_time")
        
        col2.markdown("**Model Name:** Conditional Variational Autoencoder (CVAE) (Designed DNN)")
        col2.markdown("**Model Inference Time:** " + str(inference_time))