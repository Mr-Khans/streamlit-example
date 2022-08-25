# from ast import If, main
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import os
from PIL import Image


"""
# Welcome to Streamlit!

This code about testing ImageHash
"""


list_score = []
list_mask = []
list_all = []

def smart_crop(img):
    #gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gry,(3,3), 0)
    #th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)[1]
    coords = cv2.findNonZero(img)
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
    size = (200, 300)
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Img_1"
    ws["B1"] = "Img_2"
    ws["C1"] = "Filename"
    ws["D1"] = "AVG_HASH" + str(hash_count)+ 'x' + str(hash_count) + " L"
    ws["E1"] = "d_hash_v"
    ws["F1"] = "d_hash"
    ws["G1"] = "p_hash"
    ws["H1"] = "a_hash"
    for folder_all_mask in glob.glob(path_all_mask):
        
        score , delta_5, delta_4 = hash_check(path_one_mask, folder_all_mask,hash_count)
        list_score.append(score)
        list_mask.append(folder_all_mask)
        insert_row(ws,  path_one_mask, folder_all_mask, os.path.basename(folder_all_mask),score,delta_5, delta_4, delta_3, delta_2, size=size)
    wb.save('/content/result/new_test_L' + str(hash_count)+ 'x' + str(hash_count) +'.xlsx')
    wb.close()

def load_image(image_file):
	img = Image.open(image_file)
	return img


if __name__ == '__main__':  
    uploaded_files = st.file_uploader("Choose a Image file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.image(bytes_data, caption='Load image')
        st.write("filename:", uploaded_file.name)
        
        #st.write(bytes_data)
