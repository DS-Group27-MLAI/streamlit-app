import streamlit as st
from classification import classification_inference
from anomaly_detection import anomaly_detection_inference
PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

def main():
  st.title("Image Anomaly Detection")
  menu = ['Video', "About", "Classification Inference", "Anomaly Detection Inference"]
  choice = st.sidebar.selectbox('Menu', menu)

  if choice == "Video":
    st.video(open("PNG_9048.mp4", 'rb'), format='video/mp4', start_time=0)

  if choice == "Classification Inference":
    classification_inference()

  if choice == "Anomaly Detection Inference":
    anomaly_detection_inference()
    
if __name__ == "__main__":
  
  main()