from keras.models import load_model
import cv2
from glob import glob
import streamlit as st
import numpy as np
from utils import prepare_image_from_bytes

anomaly_images_test = glob('images/anomaly/*')
normal_images_test = glob('images/normal/*')

classification_model_path = 'models/model_best_weights_classification_1.h5'

model = load_model(classification_model_path)


def load_image(label, anomaly=True):
    if anomaly:
        image = cv2.imread(anomaly_images_test[label], cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(normal_images_test[label], cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    test_X = cv2.resize(gray, (224, 224))

    return test_X


def infer_model(model, image):
    y_pred = model.predict((image.reshape(-1, 224, 224)) / 255)
    return True if y_pred[0] > 0.5 else False


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

    st.subheader("Please upload the image in RGB Format")
    st.text("Send the model to the server and detect whether the image is \nnormal/anomaly from the best model")
    uploaded_file = st.file_uploader(label="Choose a file for Anomaly Detection", type='jpeg')
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        image = prepare_image_from_bytes(st, bytes_data)
        result = infer_model(model, image)
        if result:
            st.text("Anomaly Image")
        else:
            st.text("Normal Image")
