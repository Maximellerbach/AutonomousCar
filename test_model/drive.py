import os

import cv2
import numpy as np

from custom_modules import architectures, serial_command

# serialport = '/dev/ttyS0'
# os.system('sudo chmod 0666 {}'.format(serialport))
# ser = serial_command.control(serialport)

wi = 160
he = 120

basedir = os.path.dirname(os.path.abspath(__file__))
model = architectures.safe_load_model(
    os.path.normpath(f'{basedir}/models/test.h5'))
architectures.apply_predict_decorator(model)


# dico = [10, 8, 6, 4, 2]
# dico_save = [3, 5, 7, 9, 11]
# dico_speed = [1, 0.9, 0.8, 0.9, 1]

cap = cv2.VideoCapture(0)
# ser.ChangeMotorA(2)

while(True):
    try:

        _, cam = cap.read()

        # PREPARE IMAGE FOR AI's INPUT SHAPE
        img = cv2.resize(cam, (wi, he))/255
        img_pred = np.expand_dims(img, axis=0)

        # PREDICT
        predicted = model.predict(img_pred)
        print(predicted)

        # ser.ChangePWM(speed)
        # ser.ChangeDirection(dico[pred])

        # SAVE FRAME
        # cv2.imwrite('../../../image_course/' +
        #             str(dico_save[pred])+'_'+str(time.time())+'.png', cam)

    except Exception as e:
        print(e)


cap.release()
