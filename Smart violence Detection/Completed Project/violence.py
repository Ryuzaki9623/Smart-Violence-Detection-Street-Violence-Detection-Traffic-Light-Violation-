from __future__ import absolute_import
from __future__  import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageEnhance
from imutils.object_detection import non_max_suppression
import imutils
from tkinter import filedialog as fd
import tkinter

from yolo3.yolo import YOLO

from tools import processing
from tools import generate_detections as gdet
from tools.processing import extract_parts
from tools.coord_in_box import coordinates_in_box,bbox_to_fig_ratio

from deepsort import nn_matching
from deepsort.detection import Detection
from deepsort.tracker import Tracker

from models.openpose_model import pose_detection_model
from config.config_reader import config_reader

from training.data_preprocessing import batch,generate_angles
from keras.models import load_model

from localfiletesting import *

model = Det_Model(tf,wight='fightw.hdfs')

root = tkinter.Tk()
root.withdraw()
file_path = None
file_list=[]



try:
    file_path = fd.askopenfilenames(title="Select video file(s)", filetypes=[("Video Files", "*.mp4 *.avi")])
    if not file_path: # if no file selected
        raise ValueError("No file selected") # raise a ValueError
except ValueError as e:
    print(e)
except Exception as e:
    print("An error occurred:", e)

if file_path:
    # Convert the selected file(s) to a list
    if isinstance(file_path, tuple):
        file_list = list(file_path)
    else:
        file_list = [file_path]
    # Loop through each file and process it
    for V_test in file_list:
        cap = cv2.VideoCapture(V_test)
        i = 0
        j = 0
        k=0
        frames = np.zeros((30, 160, 160, 3), dtype=float)
        old = []
        truecount = 0
        crop_idx = 0
        latest_crop_idx = 0
        imagesaved=0
        sendAlert = 0
        ysdatav2 = np.zeros((1, 30, 160, 160, 3), dtype=float)
        ysdatav2[0][:][:] = frames
        prediction = pred_fight(model,ysdatav2,acuracy=0.96)

        while(True):
            ret, frame = cap.read()
            if frame is not None:
                frame_copy = frame.copy()
            else:
                continue
            # describe the type of font to be used 
            font = cv2.FONT_HERSHEY_SIMPLEX
            #display the text on every frame
            text_color = (0, 255, 0) #Green
            label = prediction[0]
            if label: # Violence
                text_color = (0, 0, 255) # red
                truecount = truecount + 1
                
            else:# No Violence
                text_color = (0, 255, 0)
            text = "Violence: {}".format(label)
            cv2.putText(frame, text, (35, 50), font,1.25, text_color, 3)
            if i > 29:
                ysdatav2 = np.zeros((1,30,160,160, 3), dtype=float)
                ysdatav2[0][:][:] = frames
                prediction = pred_fight(model,ysdatav2,acuracy=0.96)
                if label == True:
                    cv2.imshow('video', frame)
                    print('Violence detected here ...')
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    vio = cv2.VideoWriter("D:/Smart violence Detection/violence_footage/violation-"+str(j)+".avi", fourcc, 10.0, (fwidth,fheight))                #vio = cv2.VideoWrite"./videos/output-"+str(j)+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (300, 400))
                    for frameinss in old:
                        vio.write(frameinss)
                    vio.release()
                i = 0
                j += 1
                frames = np.zeros((30, 160, 160, 3), dtype=float)
                old = []
            else:
                try:
                    frm = resize(frame,(160,160,3))
                    old.append(frame)
                    fshape = frame.shape
                    fheight = fshape[0]
                    fwidth = fshape[1]
                    frm = np.expand_dims(frm,axis=0)
                    if(np.max(frm)>1):
                        frm = frm/255
                    frames[i][:] = frm

                    i+=1
                    cv2.imshow('video', frame)
                except:
                    pass
                    if cv2.waitKey(1):
                        break

            if(truecount == 40):
                if(imagesaved == 0):
                    if(label):
                        k+=1
                        filename = 'violence-' + str(k) + '.jpg'
                        cv2.imwrite(filename,frame)
                        imagesaved = 1
                          
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

        cv2.destroyAllWindows()