from PIL import Image
import numpy as np 
import streamlit as st 

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png', 'jpeg'])

# Checking the Format of the page
if uploadFile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img = load_image(uploadFile)
    st.image(img)
    st.write("Image Uploaded Successfully")
else:
    st.write("Make sure you image is in JPG/PNG Format.")