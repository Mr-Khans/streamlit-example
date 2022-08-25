# from ast import If, main
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import os
import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt



"""
# Welcome to Streamlit!

This code about testing ImageHash
"""


list_score = []
list_mask = []
list_all = []

def smart_crop(img):
    gry = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gry,(3,3), 0)
    th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)[1]
    coords = cv2.findNonZero(th)
    x,y,w,h = cv2.boundingRect(coords)
    image = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 0)
    crop_img = image[y:y+h, x:x+w]
    crop = Image.fromarray(crop_img)
    crop = crop.convert("RGB")
    w, h = crop.size
    print(str(path))
    result_resize = float(h/w)
    if result_resize < 1.675:
        up_points = (400,400)
        crop_imag = crop.resize(up_points)
    else:
        up_points = (200,400)
        crop_imag = crop.resize(up_points)
    return crop_imag

def path_cut(path_to_file):
  path_name = os.path.normpath(path_to_file)
  text_path = str(path_name.split(os.sep)[3] + "_" + path_name.split(os.sep)[4])
  return text_path


def get_img(image, size=(100, 100)):
    img = Image.open(image)
    if size:
        img = img.resize(size)
    temp = BytesIO()
    img.save(temp, format="png")
    temp.seek(0)
    return Image.open(temp)





def hash_check_4(path_1,path_2,hash_size):
    img1 = smart_crop(path_1)
    img2 = smart_crop(path_2)


    hash_d = imagehash.dhash(img1,hash_size)
    otherhash_d = imagehash.dhash(img2,hash_size)
    delta_4 = hash_d -  otherhash_d

    hash_dhash_vertical = imagehash.dhash_vertical(img1,hash_size)
    otherhash_dhash_vertical = imagehash.dhash_vertical(img2,hash_size)
    delta_5 = hash_dhash_vertical -  otherhash_dhash_vertical

    av = ( delta_5 + delta_4)/2 
    print("AVG: ",av)
    return av, delta_5, delta_4

def check_masks_l(path_one_mask, path_all_mask,hash_count):
    for folder_all_mask in glob.glob(path_all_mask):
        
        score , delta_5, delta_4 = hash_check_4(path_one_mask, folder_all_mask,hash_count)
        list_score.append(score)
        list_mask.append(folder_all_mask)


def load_image(image_file):
	img = Image.open(image_file)
	return img

def azure_result(image_name, image):
    img = Image.open(image)
    img2 = img.crop((bbox[-1][0], bbox[-1][1], bbox[-1][4], bbox[-1][5]))


if __name__ == '__main__':  
    uploaded_files = st.file_uploader("Choose a Image file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.image(bytes_data, caption='Load image')

        st.image(smart_crop(bytes_data), caption = "crop image")


        image = Image.open(uploaded_file.read())
        image = smart_crop(image)
        st.image(image, caption = "crop image")