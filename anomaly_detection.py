from keras.models import load_model
import cv2
from glob import glob
import streamlit as st
import numpy as np
from utils import prepare_image_from_bytes

anomaly_images_test = glob('images/anomaly/*')
normal_images_test = glob('images/normal/*')

anomaly_detection_model_path = 'models/model_best_weights_anomaly_vae.h5'

model = load_model(anomaly_detection_model_path)


def load_image(label, anomaly=True):
    if anomaly:
        image = cv2.imread(anomaly_images_test[label], cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(normal_images_test[label], cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    test_X = cv2.resize(gray, (56, 56))

    return test_X


def infer_model(model, image):
    y_pred = model.predict(image.reshape(-1, 56*56))
    return y_pred.reshape(56, 56)


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

    st.subheader("Please upload the image in RGB Format")
    st.text("Send the model to the server and detect whether the image is \nnormal/anomaly from the best model")
    st.text("Please upload the images to get another image output, so that SSIM Loss can be \nevaluated")
    uploaded_file = st.file_uploader(label="Choose a file for Anomaly Detection", type='jpeg')
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        image = prepare_image_from_bytes(st, bytes_data, (56, 56))
        result = infer_model(model, (image-127.5) / 127.5)
        result = result*127.5 + 127.5
        st.image(np.clip(result.astype(np.int), a_min=0, a_max=255), caption='Output Image from network')
