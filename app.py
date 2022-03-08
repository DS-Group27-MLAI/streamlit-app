import streamlit as st
import home
import readme
import members
import about
from multiapp import MultiApp
app = MultiApp()
st.sidebar.title('Navigation')
st.sidebar.markdown('Select to Navigate')
# Add all your application here
app.add_app("Home", home.app)
app.add_app("Read Me", readme.app)
app.add_app("About", about.app)
app.add_app("Members", members.app)
# The main app
app.run()
