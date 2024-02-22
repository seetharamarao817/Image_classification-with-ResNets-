import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import requests
from pipelines.prediction_pipeline import PredictionPipeline

# Define class labels
classes = ['bird', 'ship', 'deer', 'horse', 'airplane', 'automobile', 'frog', 'dog', 'cat', 'truck']

# Streamlit UI
st.title("Image Classification with ResNet11")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction, confidence = predict(image)
    st.write(f"Prediction: {classes[prediction]} with a probability of {confidence:.2f}%")
