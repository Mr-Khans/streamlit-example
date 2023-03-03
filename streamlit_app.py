from PIL import Image
import numpy as np 
import streamlit as st 
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import numpy as np
from scipy.spatial import distance
import os
import glob
import cv2

#init paremeters
metric = 'cosine'
#accuracy - [+/- 0.01]
MIN_VALUE = 0.212076
#load_model
model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

#shape for efficientnet_lite
IMAGE_SHAPE = (224, 224)

#load layer and model
layer = hub.KerasLayer(model_url)
model = tf.keras.Sequential([layer])

# Function to Read and Manipulate Images
def load_image(img: str) -> np.ndarray:
    """
    Reads and resizes the input image
    
    Args:
    img: The image file name
    
    Returns:
    The resized image as a numpy array
    """
    im = Image.open(img).convert('L').resize(IMAGE_SHAPE)
    image = np.array(im)
    return image


# Function to Extract Embeddings for Image
def extract(file: str) -> np.ndarray:
    """
    Extracts embeddings for the given image file
    
    Args:
    file: The image file name
    
    Returns:
    The flattened feature vector as a numpy array
    """
    file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
    file = np.stack((file,)*3, axis=-1)
    file = np.array(file)/255.0
    embedding = model.predict(file[np.newaxis, ...])
    vgg16_feature_np = np.array(embedding)
    flattended_feature = vgg16_feature_np.flatten()
    return flattended_feature


def image_diff(pic_1: str, pic_2: str) -> float:
    """
    Calculates the difference between two images
    
    Args:
    pic_1: The first image file name
    pic_2: The second image file name
    
    Returns:
    The distance between the two images
    """
    img_1 = extract(pic_1)
    img_2 = extract(pic_2)
    dc = distance.cdist([img_1], [img_2], metric)[0]
    return dc



# Streamlit App
st.header("TEST mask")
# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload mask for test", type=['jpg', 'png', 'jpeg'])
uploadFile_ = st.file_uploader(label="Upload mask in dataset", accept_multiple_files=False, type=['jpg', 'png', 'jpeg'])


# Checking the Format of the page
if uploadFile is not None:
    # Perform your Manipulations (In my Case applying Filters)
    img_1 = load_image(uploadFile)
    st.image(img_1)
    st.write("Image Uploaded Successfully")
else:
    st.write("Make sure your image is in JPG/PNG/JPEG Format.")


# Checking the Format of the page
if uploadFile_ is not None:
    # Perform your Manipulations (In my Case applying Filters)
    img_2 = load_image(uploadFile_)
    st.image(img_2)
    st.write("DATASET Uploaded Successfully")
else:
    st.write("Make sure your image is in JPG/PNG/JPEG Format.")


if st.button('Result'):
    result = image_diff(uploadFile, uploadFile_)
    st.write("File: ", str(uploadFile_.name), str(result))
    st.write(str(uploadFile_))
    if MIN_VALUE > result:
        st.write("GOOD MASK!!!!: ", str(result))
    else:
        st.write("NEED NEW MASK: ", str(result))

#proof
else:
    st.write('LOAD TWO IMAGES')

if st.button('Clear cache'):
    st.legacy_caching.caching.clear_cache()
    # with st.form("my-form", clear_on_submit=True):
    #     file = st.file_uploader("FILE UPLOADER")
    #     submitted = st.form_submit_button("UPLOAD!")
    uploadFile,uploadFile_ = None,None
    #print()
else:
    st.write('PLS press F5')

