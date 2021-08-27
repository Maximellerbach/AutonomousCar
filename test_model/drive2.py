import os
import sys

import cv2
import time

from custom_modules import serial_command2, architectures
from custom_modules.datasets import dataset_json

serialport = '/dev/ttyUSB0'
os.system('sudo chmod 0666 {}'.format(serialport))
ser = serial_command2.control(serialport)

MAXTHROTTLE = 0.3
wi = 160
he = 120

Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
input_components = []

basedir = os.path.dirname(os.path.abspath(__file__))
model = architectures.load_model(
    os.path.normpath(f'{basedir}/models/auto_label5.h5'))
architectures.apply_predict_decorator(model)

cap = cv2.VideoCapture(0)
ret, img = cap.read()  # read the camera once to make sure it works
assert ret is True

print("Starting mainloop")

while(True):
    try:
        _, cam = cap.read()
        img = cv2.resize(cam, (wi, he))

        memory = {}
        memory['direction'] = 0
        memory['speed'] = 0
        memory['throttle'] = 0.18
        memory['time'] = time.time()

        to_pred = Dataset.make_to_pred_annotations(
            [img], [memory], input_components)

        # PREDICT
        predicted, dt = model.predict(to_pred)
        memory['direction'] = predicted[0]['direction']

        print(predicted, dt)

        ser.ChangeAll(memory['direction'], MAXTHROTTLE*memory['throttle'])

    except Exception as e:
        print(e)

    except KeyboardInterrupt:
        sys.exit()


cap.release()
