import os
import time

import cv2
import numpy as np
from keras.models import load_model

import SerialCommand

os.system('sudo chmod 0666 /dev/ttyS0')

ser = SerialCommand.control("/dev/ttyS0")

wi = 160
he = 120

speed = int(input('speed: '))

dico = [10,8,6,4,2]
dico_save = [3,5,7,9,11]
#model= load_model(os.path.dirname(__file__) + os.path.normpath("\\vroum.h5"))
model = load_model("nofilter_ironcar.h5")

cap = cv2.VideoCapture(0)

ser.ChangeMotorA(2)

i = 0
while(True):
    try:
        
        _, cam= cap.read()
        i+=1

        ''' # color filter
        hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
        
        lower = np.array([31,60,60])
        upper = np.array([50,255,255])
        
        #FILTERING COLOR WITH COLOR RANGE
        mask = cv2.inRange(hsv, lower, upper)
        image_tab = cv2.bitwise_and(cam,cam, mask= mask)
        '''

        #PREPARE IMAGE FOR AI's INPUT SHAPE
        img = cv2.resize(cam,(wi,he))
        img_pred = np.expand_dims(img, axis= 0)

        #PREDICT
        predicted = np.argmax(model.predict(img_pred))
        predicted = int(predicted)
        prediction = dico[predicted] 

        
        #APPLY DIR AND SPEED
        if predicted == 0 or predicted == 4:
            actual_speed = speed +5
        elif predicted == 1 or predicted == 3:
            actual_speed = speed +3
        elif predicted == 2:
            actual_speed = speed

        ser.ChangePWM(actual_speed)
        ser.ChangeDirection(prediction)
         
        #SAVE FRAME
        #cv2.imwrite('../../../image_course/'+str(dico_save[predicted])+'_'+str(time.time())+'.png',img)
        
    except:
        print("error in program's loop",i)


cap.release()
