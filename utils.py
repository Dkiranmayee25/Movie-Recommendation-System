import streamlit as st
import base64

def set_background_with_overlay(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        position: relative;
    }}
    
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.4);  /* Dark overlay */
        z-index: -1;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
