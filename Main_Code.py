######################################## pwd
import sys
pwd=sys.path[0]
########################################

######################################## Imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
###
import tkinter as tk
###
import cv2
########################################

######################################## Import from folder
folder_name='Functions'
path_name_new=pwd + '\\' + folder_name
sys.path.append(path_name_new)
##### Import Files
import Main_Func, Define
########################################

################################################################################
################################################################################
################################################################################

########## load model

### load SVM 1
file_name=pwd+'\\'+ 'Model_MLR' +'.obj'
filehandler  = open(file_name,'rb')
model = pickle.load(filehandler)
filehandler.close()

########## Instructions
print('--------------------')
print('Instructions:')
print('Right Click - clear')
print('Double Left Click - exit program')

################################################################################ functions

########## inputs 
pixel_window=Define.Parameters.pixel_window
pixel_norm=Define.Parameters.pixel_norm
draw_radius=Define.Parameters.draw_radius
#

### parameters
img=[]
features=[]
counter=0 
flag_exit=0
###
drawing = False 

########## draw circle
def draw_circle(event,x,y,flags,param):
    global drawing, img, features
    global counter, flag_exit

    if event == cv2.EVENT_LBUTTONDBLCLK: # double click for exit and save data 
        flag_exit=1

    if event == cv2.EVENT_RBUTTONDOWN: # click for clear img
        ### new window
        img=np.full((pixel_window,pixel_window,3),0,np.float64)
      
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),draw_radius,(255,255,255),-1)
                
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    
############################################################

########## Print delay time
show_time=40 
show_time_i=0

########## main draw
def Main_Draw():
    global show_time_i, show_time 
    global flag_exit
    global img
    global model
    
    ### new window
    img=np.full((pixel_window,pixel_window,3),0,np.float64) # new window

    ### draw loop
    while(flag_exit==0): 
        ### show window
        cv2.imshow('MLR',img)
        cv2.namedWindow('MLR')
        cv2.setMouseCallback('MLR',draw_circle) # call draw function

        if show_time_i==show_time:
            ### picture proccesing
            pic=Main_Func.Picture_Proccess(img, pixel_norm)

            ### predict
            prediction=Prediction_Func(pic)
            ###
            show_time_i=0
            print("---")
            print(prediction)

        ### update show counter   
        show_time_i+=1

        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    print("----------------------------------------")
    flag_exit=0
##########

########## predict function
def Prediction_Func(pic):
    global model
    flat_pic = pic.reshape(pic.size, ) # flatten
    prediction=model.Predict(flat_pic)
    prediction= Main_Func.Classify_Class(prediction)
    ###
    return prediction

############################## main
Main_Draw()



