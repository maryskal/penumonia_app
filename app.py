import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
import prepare_img as pr_img


def round_fun(float_list):
    for i, fl in enumerate(float_list):
        float_list[i] = '{}%'.format(round(fl*100, 2))
    return float_list


def img_to_array(image_file):
    img = Image.open(image_file)
    return np.array(img)


if __name__ == '__main__':
    st.markdown("<h1 style='text-align: center; color: gray;'>PNEUMONIA DIAGNOSIS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center'>Load your Chest X-Ray image and the model will calculate its pneumonia probability</p>", unsafe_allow_html=True)
   
   # Load model and image
    model = load_model('./models/pneumonia_classification_model.h5')
    image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if image_file is not None:
        # Display image in the center
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image(image_file)
        with col3:
            st.write(' ')
        
        # Get prepared image and predict
        prepared_img = pr_img.get_prepared_img(img_to_array(image_file), 512, True)
        results = model.predict(prepared_img)

        # Display results
        results = round_fun(results[0].tolist())
        col4, col5, col6 = st.columns(3)
        col4.metric("Normal", results[0])
        col5.metric("Mild", results[1])
        col6.metric("Moderate-Severe", results[2])
        st.markdown('''
        <style>
        /*center metric label*/
        [data-testid="stMetricLabel"] > div:nth-child(1) {
            justify-content: center;
        }

        /*center metric value*/
        [data-testid="stMetricValue"] > div:nth-child(1) {
            justify-content: center;
        }
        </style>
        ''', unsafe_allow_html=True)