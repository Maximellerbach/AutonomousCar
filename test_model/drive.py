import os

import cv2
import time

from custom_modules import serial_command, architectures, drive_utils
from custom_modules.datasets import dataset_json

# serialport = '/dev/ttyS0'
# os.system('sudo chmod 0666 {}'.format(serialport))
# ser = serial_command.control(serialport)

wi = 160
he = 120

Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
input_components = [1]

basedir = os.path.dirname(os.path.abspath(__file__))
model = architectures.load_model(
    os.path.normpath(f'{basedir}/models/gentrck_sim1_working.h5'))
# architectures.apply_predict_decorator(model)


cap = cv2.VideoCapture(0)
# ser.ChangeMotorA(2)

while(True):
    try:
        _, cam = cap.read()

        # PREPARE IMAGE FOR AI's INPUT SHAPE
        img = cv2.resize(cam, (wi, he))/255

        # annotation template with just what is needed for the prediction
        annotation = {
            'direction': 0,
            'speed': 10,
            'throttle': 0,
            'time': 0
        }
        annotation_list = drive_utils.dict2list(annotation)

        to_pred = Dataset.make_to_pred_annotations(
            [img], [annotation_list], input_components)

        # PREDICT
        st = time.time()
        predicted = model.predict(to_pred)
        et = time.time()
        print(predicted, et-st)

        cv2.imshow('img', img)
        cv2.waitKey(1)

        # ser.ChangePWM(speed)
        # ser.ChangeDirection(dico[pred])

        # SAVE FRAME
        # cv2.imwrite('../../../image_course/' +
        #             str(dico_save[pred])+'_'+str(time.time())+'.png', cam)

    except Exception as e:
        print(e)


cap.release()
