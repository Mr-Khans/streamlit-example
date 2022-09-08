from PIL import Image
import numpy as np 
import streamlit as st 

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