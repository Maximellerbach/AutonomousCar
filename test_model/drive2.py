import os

import cv2
import time

from custom_modules import serial_command2, architectures, drive_utils
from custom_modules.datasets import dataset_json

serialport = '/dev/ttyUSB0'
os.system('sudo chmod 0666 {}'.format(serialport))
ser = serial_command2.control(serialport)

MAXTHROTTLE = 0.4
wi = 160
he = 120

Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
input_components = [1]

basedir = os.path.dirname(os.path.abspath(__file__))
model = architectures.load_model(
    os.path.normpath(f'{basedir}/models/test_home.h5'))
architectures.apply_predict_decorator(model)

cap = cv2.VideoCapture(0)

prev_throttle = 0
while(True):
    try:
        _, cam = cap.read()
        img = cv2.resize(cam, (wi, he))

        # annotation template with just what is needed for the prediction
        annotation = {
            'direction': 0,
            'speed': prev_throttle,
            'throttle': 0,
            'time': time.time()
        }
        annotation_list = drive_utils.dict2list(annotation)

        to_pred = Dataset.make_to_pred_annotations(
            [img/255], [annotation_list], input_components)

        # PREDICT
        predicted, dt = model.predict(to_pred)
        predicted = predicted[0]
        steering = predicted['direction'][0]
        throttle = predicted['throttle'][0]

        print(predicted)
        # print(dt)

        # cv2.imshow('img', img)
        # cv2.waitKey(1)

        ser.ChangeAll(steering, throttle*MAXTHROTTLE)
        prev_throttle = predicted['throttle'][0]

        # SAVE FRAME
        # Dataset.save_img_and_annotation(img, annotation)

    except Exception as e:
        print(e)


cap.release()
