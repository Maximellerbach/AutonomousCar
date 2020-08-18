import os
import time

import cv2
import numpy as np
from keras.models import load_model


from custom_modules import SerialCommand

serialport = '/dev/ttyS0'
os.system('sudo chmod 0666 {}'.format(serialport))
ser = SerialCommand.control(serialport)

wi = 160
he = 120

model = load_model("lightv6_mix.h5")


dico = [10, 8, 6, 4, 2]
dico_save = [3, 5, 7, 9, 11]
dico_speed = [1, 0.9, 0.8, 0.9, 1]
speed = int(input('speed: '))

cap = cv2.VideoCapture(0)
ser.ChangeMotorA(2)

while(True):
    try:

        _, cam = cap.read()

        # PREPARE IMAGE FOR AI's INPUT SHAPE
        img = cv2.resize(cam, (wi, he))/255
        img_pred = np.expand_dims(img, axis=0)

        # PREDICT
        predicted = model.predict(img_pred)
        pred = np.argmax(predicted[0])
        # predicted = ((model.predict(img_pred)+1)*4)+3

        ser.ChangePWM(speed)
        ser.ChangeDirection(dico[pred])

        # SAVE FRAME
        cv2.imwrite('../../../image_course/' +
                    str(dico_save[pred])+'_'+str(time.time())+'.png', cam)

    except Exception as e:
        print(e)


cap.release()
