import os
import time
#import IMUSensor
#from simple_pid import PID

import cv2
import numpy as np
from keras.models import load_model
from threading import Thread
import SerialCommand

serialport = '/dev/ttyS0'
os.system('sudo chmod 0666 {}'.format(serialport))
ser = SerialCommand.control(serialport)

wi = 160
he = 120

speed = int(input('speed: '))

dico = [10,8,6,4,2]
dico_save = [3,5,7,9,11]
dico_speed = [1, 0.9, 0.8, 0.9, 1]
#model= load_model(os.path.dirname(__file__) + os.path.normpath("\\vroum.h5"))
model = load_model("lightv1_robo.h5")

'''
vel_sensor = IMUSensor.sensor(10, 1000)
thread = Thread(target = vel_sensor.loop)
thread.start()

pid = PID(1, 0.5, 0.1)
pid.setpoint = 0.1 #targeted speed
pid.output_limits = (0, 1)
v = vel_sensor.get_vel()
'''

cap = cv2.VideoCapture(0)
ser.ChangeMotorA(2)

while(True):
    try:
        
        _, cam= cap.read()

        #PREPARE IMAGE FOR AI's INPUT SHAPE
        img = cv2.resize(cam,(wi,he))/255
        img_pred = np.expand_dims(img, axis= 0)

        #PREDICT
        predicted = model.predict(img_pred)
        pred = np.argmax(predicted[0])
        # predicted = ((model.predict(img_pred)+1)*4)+3

        '''
        v = vel_sensor.get_vel()
        control = pid(v)
        control = int(control*100)
        print(control, v)
        '''

        ser.ChangePWM(speed)
        ser.ChangeDirection(dico[pred])
        
        # SAVE FRAME
        cv2.imwrite('../../../image_course/'+str(dico_save[pred])+'_'+str(time.time())+'.png',cam)
        #cv2.imwrite('../../../image_course/'+str((lab-2)/2)+str(time.time())+'_'+str(time.time())+'.png',img) # reg

    except Exception as e:
        print(e)


cap.release()
