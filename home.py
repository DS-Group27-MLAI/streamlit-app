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
#def main():
 #   st.title("Image Anomaly Detection")
    # menu = ['Video', "About", "Classification Inference", "Anomaly Detection Inference"]
    # choice = st.sidebar.selectbox('Menu', menu)


#PAGES = {
    # "Home": home,
 #   "Readme": readme,
  #  "About": about,
   # "Members": members
#}#

#selection = st.sidebar.radio("Go To", list(PAGES.keys()))
#page = PAGES[selection]
#page.app()
# add_home = st.sidebar.header("[Home](https://wandererluzon.wixsite.com/my-site)")
# add_readme = st.sidebar.header("[Read Me](upload://readme.py)")
# add_about = st.sidebar.header("[About](https://wandererluzon.wixsite.com/my-site)")
# add_members = st.sidebar.header("[Members](https://wandererluzon.wixsite.com/my-site)")




# if choice == "Video":
#    st.video(open("PNG_9048.mp4", 'rb'), format='video/mp4', start_time=0)

# if choice == "Classification Inference":
#    classification_inference()
# if choice == "Anomaly Detection Inference":
#   anomaly_detection_inference()


#if __name__ == "__main__":
    # <<<<<<< HEAD
    # main()
    # =======
    #main()