# from ast import If, main
from collections import namedtuple
import altair as alt
import streamlit as st
import os
import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np
from skimage.metrics import structural_similarity
import phasepack.phasecong as pc
import rasterio
from enum import Enum
from io import BytesIO, StringIO
from typing import Union


"""
# Welcome to Streamlit!

## This code about find same images
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


def read_image(name):
  image = st.file_uploader("Upload an "+ name, type=["png", "jpg", "jpeg"])
  if image:
    im = Image.open(image)
    im.filename = image.name
    return im


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


def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
    # shape of the image should be like this (rows, cols, bands)
    # Please note that: The interpretation of a 3-dimension array read from rasterio is: (bands, rows, columns) while
    # image processing software like scikit-image, pillow and matplotlib are generally ordered: (rows, columns, bands)
    # in order efficiently swap the axis order one can use reshape_as_raster, reshape_as_image from rasterio.plot
    msg = (
        f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
        f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}"
    )

    assert org_img.shape == pred_img.shape, msg


def rmse(org_img: np.ndarray, pred_img: np.ndarray, max_p: int = 4095) -> float:
    """
    Root Mean Squared Error
    Calculated individually for all bands, then averaged
    """
    _assert_image_shapes_equal(org_img, pred_img, "RMSE")

    rmse_bands = []
    
    for i in range(org_img.shape[2]):
        dif = np.subtract(org_img[:, :, i], pred_img[:, :, i])
        m = np.mean(np.square(dif / max_p))
        s = np.sqrt(m)
        rmse_bands.append(s)

    return np.mean(rmse_bands)


def read_image(path: str):
    return Image.open(path)


def evaluation(org_img_path: str, pred_img_path: str):
    org_img = read_image(org_img_path)
    pred_img = read_image(pred_img_path)
    np.seterr(divide = 'ignore') 
    width, height = 100,100
    dim = (width, height)
# resize image
    resized_1 = cv2.resize(org_img, dim, interpolation = cv2.INTER_AREA)
    resized_2 = cv2.resize(pred_img, dim, interpolation = cv2.INTER_AREA)
    out_value = float(rmse(resized_1, resized_2))
    output = out_value
    #print(output)
    return output

def evaluation_(org_img, pred_img):
    np.seterr(divide = 'ignore') 
    width, height = 100,100
    dim = (width, height)
# resize image
    #resized_1 = cv2.resize(org_img, dim, interpolation = cv2.INTER_AREA)
    #resized_2 = cv2.resize(pred_img, dim, interpolation = cv2.INTER_AREA)
    out_value = float(rmse(org_img, pred_img))
    output = out_value
    #print(output)
    return output

# def rotate_img(path: str):
#   img = cv2.imread(path)
#   out=cv2.transpose(img)
#   out=cv2.flip(out,flipCode=90)
#   width, height = 100,100
#   dim = (width, height)
#   out=cv2.transpose(img)
#   out=cv2.flip(out,flipCode=0)
#   resized = cv2.resize(out, dim, interpolation = cv2.INTER_AREA)
#   cv2.imwrite("rotated_90.jpg", resized) 
#   return(resized)


#result = (evaluation("/content/1.jpg","/content/_0101_01_l.png"))
#print(result)

def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright


def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img


def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr


def main_loop():
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to play with Image filters!")
    st.text("We use OpenCV and Streamlit for this demo")

    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)
    st.text("Result RMSE")
    st.text(float(evaluation(image_file,image_file)))

    processed_image = blur_image(original_image, blur_rate)
    processed_image = brighten_image(processed_image, brightness_amount)

    if apply_enhancement_filter:
        processed_image = enhance_details(processed_image)

    st.text("Original Image vs Processed Image")
    st.image([original_image, processed_image])


if __name__ == '__main__':
    main_loop()