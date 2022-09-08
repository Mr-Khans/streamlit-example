

from PIL import Image
import numpy as np 
import streamlit as st 
import tensorflow as tf

import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os
import glob
import cv2



#init paremeters
metric = 'cosine'

#load_model
model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

#shape for efficientnet_lite
IMAGE_SHAPE = (224, 224)

#load layer and model
layer = hub.KerasLayer(model_url)
model = tf.keras.Sequential([layer])

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload mask for test", type=['jpg', 'png', 'jpeg'])

# Checking the Format of the page
if uploadFile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img_1 = load_image(uploadFile)
    st.image(img_1)
    st.write("Image Uploaded Successfully")
else:
    st.write("Make sure you image is in JPG/PNG Format.")


uploadFile_ = st.file_uploader(label="Upload mask in dataset", type=['jpg', 'png', 'jpeg'])

# Checking the Format of the page
if uploadFile_ is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img_2 = load_image(uploadFile_)
    st.image(img_2)
    st.write("Image Uploaded Successfully")
else:
    st.write("Make sure you image is in JPG/PNG Format.")


