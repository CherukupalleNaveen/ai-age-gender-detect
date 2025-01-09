import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import MeanAbsoluteError

import cv2

# Load the pre-trained model at the start of the app
@st.cache_resource
def load_trained_model():
    model = load_model('models/model_30epochs.h5', custom_objects={'mae': MeanAbsoluteError})
    return model

# Load the model
model = load_trained_model()

# Gender and Age dictionaries
gender_dict = {0: 'Male', 1: 'Female'}  # Updated gender dictionary

def preprocess_image(uploaded_image):
    """Preprocess the uploaded image for model prediction."""
    # Convert the uploaded image to an array
    img = image.load_img(uploaded_image, target_size=(128, 128), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image if required by the model
    return img_array

def predict_gender_and_age(img_array):
    """Predict gender and age from the preprocessed image."""
    pred = model.predict(img_array)
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    return pred_gender, pred_age

def main():
    st.title("Gender and Age Prediction from Image")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Preprocess the image
        img_array = preprocess_image(uploaded_file)
        
        # Predict gender and age
        pred_gender, pred_age = predict_gender_and_age(img_array)
        
        # Display the results
        st.markdown(f"Predicted Gender: **{pred_gender}**")
        st.markdown(f"Predicted Age: **{pred_age}**")
        
        # Show the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

if __name__ == '__main__':
    main()

