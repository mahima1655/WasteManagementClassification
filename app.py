import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import io

# Load the pre-trained model
model = load_model('plastic_waste_model.h5')

# Function to preprocess the image for prediction
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the size the model expects
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0  # Normalize pixel values to between 0 and 1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make a prediction
def predict_image(image):
    image = preprocess_image(image)
    prediction = model.predict(image)  # Predict the class
    return prediction

# Streamlit interface
st.title("Plastic Waste Classification")
st.write("Upload an image of plastic waste to classify it into categories such as Organic or Inorganic.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Make a prediction
    prediction = predict_image(image)
    
    # Output the prediction result
    if prediction[0] > 0.5:
        st.write("This is Inorganic Plastic Waste.")
    else:
        st.write("This is Organic Plastic Waste.")
    
    # Optionally display a probability score
    st.write(f"Prediction confidence: {prediction[0][0]:.2f}")


