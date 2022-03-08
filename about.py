import streamlit as st


def app():
    st.title('About')
    st.write('This Video is about the Work presentation')
    st.video(open("PNG_9048.mp4", 'rb'), format='video/mp4', start_time=0)