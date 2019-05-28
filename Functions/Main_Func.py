######################################## Imports
import numpy as np
from scipy import misc
from skimage import color
import tensorflow as tf
import Define
########################################

####################
def Classify_Class(prediction_in):
    classes=Define.Parameters.classes
    return classes[prediction_in]
####################

####################
def Picture_Proccess(img, pixel_norm):
    img = misc.imresize(img, (pixel_norm,pixel_norm)) # 3 channel RGB 28x28
    gray = color.rgb2gray(img) # 1 channel grayscale
    gray_scaled = misc.bytescale(gray, high=255, low=0) # scaled to 255
    pic = tf.keras.utils.normalize(gray_scaled, axis =1) # norm
    return pic
####################

####################
def Pack_Set(features, labels):  
    list=Init_List(2)
    list[0]=features
    list[1]=labels
    return list # list[0]= features, list[1]= labels
####################

####################
def Unpack_Set(list):
    return list[0], list[1] # return: features, labels
####################

####################    
def Count_Digits(labels):
    list_digits=Init_List(10)
    num_of_labels=np.size(labels)
    for i in range(num_of_labels):
        cur_label=labels[i]
        list_digits[cur_label]+=1
    return list_digits
####################

####################
def Init_List(n):
    list=[]
    for i in range(0,n):
        list.append(0)
    return list
####################

####################
def Get_Name_From_File_Path(file_path):
    str_path=file_path.split("/")
    file_name=str_path[len(str_path)-1]
    str_path=file_name.split(".")
    file_name=str_path[0]
    return file_name
####################


