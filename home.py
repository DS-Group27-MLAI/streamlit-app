import streamlit as st
from classification import classification_inference
from anomaly_detection import anomaly_detection_inference

PAGE_CONFIG = {"page_title": "StColab.io", "page_icon": ":smiley:", "layout": "centered"}
st.set_page_config(**PAGE_CONFIG)
def app():
    st.title("Home")
    option_selected = st.radio("Select the Image processing method",('Anomaly Detection', 'Classification of Image'))
    if option_selected == 'Anomaly Detection':
        st.write('Anomaly Detection is selected')
        anomaly_detection_inference()
    else:
        st.write("Classification of Image is selected")
        classification_inference()