

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
    im = Image.open(img).convert('L').resize(IMAGE_SHAPE)
    image = np.array(im)
    return image

#def embedding for image
def extract(file):
  file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
  #display(file)
  file = np.stack((file,)*3, axis=-1)
  file = np.array(file)/255.0
  embedding = model.predict(file[np.newaxis, ...])
  #print(embedding)
  vgg16_feature_np = np.array(embedding)
  flattended_feature = vgg16_feature_np.flatten()
  #print(len(flattended_feature))
  #print(flattended_feature)
  #print('-----------')
  return flattended_feature


def image_diff(pic_1, pic_2):
    img_1 = extract(pic_1)
    #org_img = cv2.imread(path_mask)
    #for folder_all_mask in glob.glob(mask_path_xl):
    img_2 = extract(pic_2)
    #second_img = cv2.imread(folder_all_mask)
    dc = distance.cdist([img_1], [img_2], metric)[0]
      #list_score.append(float(dc))
      #list_mask.append(folder_all_mask)
    return dc
    #   plt.subplot(121)
    #   plt.imshow(org_img)
    #   plt.subplot(122)
    #   plt.imshow(second_img)
    #   plt.show()
    #   print(folder_all_mask, float(dc))
st.header("Tast same mask")
# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload mask for test", type=['jpg', 'png', 'jpeg'])
uploadFile_ = st.file_uploader(label="Upload mask in dataset", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])
#uploadFile_ = st.file_uploader(label="Upload mask in dataset",  type=['jpg', 'png', 'jpeg'])

# Checking the Format of the page
if uploadFile is not None:
    
    # Perform your Manupilations (In my Case applying Filters)
    img_1 = load_image(uploadFile)
    st.image(img_1)
    st.write("Image Uploaded Successfully")
else:
    st.write("Make sure you image is in JPG/PNG/JPEG Format.")


# Checking the Format of the page
if uploadFile_ is not None:
    # Perform your Manupilations (In my Case applying Filters)
    #img_2 = load_image(uploadFile_)
    #st.image(img_2)
    st.write("DATASET Uploaded Successfully")
else:
    st.write("Make sure you image is in JPG/PNG/JPEG Format.")

if st.button('Result'):
    st.write(str(uploadFile_))
    st.write(len(uploadFile_))
    #for i in range(len(uploadFile_)):
    for i in range(len(uploadFile_)):
         #= uploadFile_[i]
        
        st.write(str(uploadFile_.name),": ", str(image_diff(uploadFile, uploadFile_[i].name)))
else:
    st.write('LOAD TWO IMAGES')

