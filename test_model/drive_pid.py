import os

import cv2
import time

from custom_modules import serial_command2, architectures, pid_steering
from custom_modules.datasets import dataset_json

serialport = '/dev/ttyUSB0'
os.system('sudo chmod 0666 {}'.format(serialport))
ser = serial_command2.control(serialport)

MAXTHROTTLE = 0.45
wi = 160
he = 120

pidSteering = pid_steering.SimpleSteering()

Dataset = dataset_json.Dataset(["direction", "speed", "cte", "throttle", "time"])
input_components = [1]

basedir = os.path.dirname(os.path.abspath(__file__))
model = architectures.load_model(
    os.path.normpath(f'{basedir}/models/test_home.h5'))
architectures.apply_predict_decorator(model)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

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

        to_pred = Dataset.make_to_pred_annotations(
            [img], [annotation], input_components)

        # PREDICT
        predicted, dt = model.predict(to_pred)
        predicted = predicted[0]
        cte = predicted['cte']

        steering = pidSteering.update_steering(cte)
        ser.ChangeAll(steering, 0.18)

        prev_throttle = predicted['throttle'][0]

    except Exception as e:
        print(e)


cap.release()
