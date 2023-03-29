import streamlit as st
import json
from typing import List
import requests as r
import base64
from PIL import Image
import os
from io import BytesIO
import io



ENDPOINT_URL = os.environ.get('ENDPOINT_URL')
HF_TOKEN = os.environ.get('HF_TOKEN')


def encode_image(image_path):
  with open(image_path, "rb") as i:
    b64 = base64.b64encode(i.read())
  return b64.decode("utf-8")


def predict(prompt, image_path, negative_prompt="worst quality, low quality, oil painting, historic", controlnet_type="canny_edge"):
    image = encode_image(io.BytesIO(image_path))

    # prepare sample payload
    payload = {"inputs": prompt, "image": image, "negative_prompt": negative_prompt, "controlnet_type": controlnet_type}

    # headers
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "image/png" # important to get an image back
    }

    response = r.post(ENDPOINT_URL, headers=headers, json=payload)
    img = Image.open(BytesIO(response.content))
    return img

st.title("Product Photography")

# Get user input
prompt = st.text_input("Enter a prompt:", "sneakers kept on sea sand")
image_path = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if image_path is not None:
    image_bytes = image_path.read()

# Make prediction and display result
if image_path is not None:
    st.image(image_path, caption="Uploaded Image", use_column_width=True)
    prediction = predict(prompt, image_path=image_bytes, negative_prompt="worst quality, low quality, oil painting, historic", controlnet_type="canny_edge")
    st.image(prediction, caption="Predicted Image", use_column_width=True)
else:
    st.write("Please upload an image.") 