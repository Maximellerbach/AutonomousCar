import os

import cv2
import time

from custom_modules import serial_command2, architectures, drive_utils
from custom_modules.datasets import dataset_json

serialport = '/dev/ttyUSB0'
os.system('sudo chmod 0666 {}'.format(serialport))
ser = serial_command2.control(serialport)

wi = 160
he = 120

Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
input_components = [1]

basedir = os.path.dirname(os.path.abspath(__file__))
model = architectures.load_model(
    os.path.normpath(f'{basedir}/models/gentrck_sim1_working.h5'))
architectures.apply_predict_decorator(model)

cap = cv2.VideoCapture(0)

while(True):
    try:
        _, cam = cap.read()
        img = cv2.resize(cam, (wi, he))

        # annotation template with just what is needed for the prediction
        annotation = {
            'direction': 0,
            'speed': 10,
            'throttle': 0,
            'time': time.time()
        }
        annotation_list = drive_utils.dict2list(annotation)

        to_pred = Dataset.make_to_pred_annotations(
            [img/255], [annotation_list], input_components)

        # PREDICT
        predicted, dt = model.predict(to_pred)
        predicted = predicted[0]
        for name in predicted.keys():
            annotation[name] = predicted[name][0]  # in our case we only have 1 value per output, so converting the array to a single float

        # print(annotation)
        print(dt)

        # cv2.imshow('img', img)
        # cv2.waitKey(1)

        ser.ChangePWM(0)
        ser.ChangeDirection(annotation['direction'])

        # SAVE FRAME
        # Dataset.save_img_and_annotation(img, annotation)

    except Exception as e:
        print(e)


cap.release()
