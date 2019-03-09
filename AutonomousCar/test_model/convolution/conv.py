import cv2
import imutils
import scipy as sp

from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential, load_model

import numpy as np
import h5py
from tqdm import tqdm
from glob import glob

import time
import os
import datetime

os.system('sudo chmod 0666 /dev/ttyS0')
import SerialCommand

ser = SerialCommand.control("/dev/ttyS0")

wi = 160
he = 120

speed = int(input('speed: '))

dico = [10,8,6,4,2]
dico_save = [3,5,7,9,11]
#model= load_model(os.path.dirname(__file__) + os.path.normpath("\\vroum.h5"))
model= load_model("new_conv_dropout10.h5")

cap = cv2.VideoCapture(0)

ser.ChangeMotorA(2)

while(True):
    try: 
        _, cam= cap.read()
        
        hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
        
        lower = np.array([31,60,60])
        upper = np.array([50,255,255])
        
        #FILTERING COLOR WITH COLOR RANGE
        mask = cv2.inRange(hsv, lower, upper)
        image_tab = cv2.bitwise_and(cam,cam, mask= mask)

        #PREPARE IMAGE FOR AI's INPUT SHAPE
        img = cv2.resize(image_tab,(wi,he))
        img_pred = np.expand_dims(img, axis= 0)

        #PREDICT
        predicted = np.argmax(model.predict(img_pred))
        predicted = int(predicted)
        prediction = dico[predicted] 

        #APPLY DIR AND SPEED
        if predicted == 0 or predicted == 4:
            ser.ChangePWM(speed-5)
        elif predicted == 1 or predicted == 3:
            ser.ChangePWM(speed)
        elif predicted == 2:
            ser.ChangePWM(speed+5)

        ser.ChangeDirection(prediction)
        
        #SAVE FRAME
        cv2.imwrite('../../../image_course/'+str(dico_save[predicted])+'_'+str(time.time())+'.png',img)

    except :
        print('err')

cap.release()
