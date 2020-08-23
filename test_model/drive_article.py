import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model


from custom_modules import serial_command

dico = [3, 5, 7, 9, 11]
speed = 70

model = load_model("models/model.h5")

cap = cv2.VideoCapture(0)

ser = serial_command.control("/dev/ttyS0")
ser.ChangeMotorA(2)

while(True):
    try:
        _, cam = cap.read()

        # PREPARE IMAGE
        img = cv2.resize(cam, (160, 120))
        img_pred = np.expand_dims(img, axis=0)

        # PREDICT
        predicted = np.argmax(model.predict(img_pred))
        direction = dico[predicted]

        ser.ChangePWM(speed)
        ser.ChangeDirection(direction)

        # SAVE FRAME
        # cv2.imwrite('../../../image_course/'+str(dico_save[predicted])+'_'+str(time.time())+'.png',img)

    except:
        print("error in program's loop")

ser.ChangePWM(0)
cap.release()
