import os

import cv2
import time

from custom_modules import serial_command2, architectures, memory
from custom_modules.datasets import dataset_json

serialport = '/dev/ttyUSB0'
os.system('sudo chmod 0666 {}'.format(serialport))
ser = serial_command2.control(serialport)

MAXTHROTTLE = 0.45
wi = 160
he = 120

Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
input_components = []

basedir = os.path.dirname(os.path.abspath(__file__))
model = architectures.load_model(
    os.path.normpath(f'{basedir}/models/test_home.h5'))
architectures.apply_predict_decorator(model)
memory = memory.Memory(2)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

while(True):
    try:
        _, cam = cap.read()
        img = cv2.resize(cam, (wi, he))

        memory['direction'] = 0
        memory['speed'] = 0
        memory['throttle'] = 0.18
        memory['time'] = time.time()

        to_pred = Dataset.make_to_pred_annotations(
            [img], [memory[-1]], input_components)

        # PREDICT
        predicted, dt = model.predict(to_pred)
        for key in predicted[0].keys():
            memory[key] = predicted[0][key][0]

        print(memory)
        # print(dt)

        # cv2.imshow('img', img)
        # cv2.waitKey(1)

        ser.ChangeAll(memory['direction'], memory['throttle'])
        memory.append({})

    except Exception as e:
        print(e)


cap.release()
