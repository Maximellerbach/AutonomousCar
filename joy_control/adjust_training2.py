'''
joystick button usage:
    'joy_steering': 'direction',
    'joy_throttle': 'throttle',
    'joy_brake': 'brake',
    'joy_button_a': 'overide prediction and save img'
    'joy_button_x': 'disable direction when joy_button_a is pressed'
'''
import cv2
import os
import time

import xbox
from custom_modules import serial_command2, architectures, drive_utils
from custom_modules.datasets import dataset_json


Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
input_components = [1]

dos_save = os.getcwd()+os.path.normpath("/recorded/")
if not os.path.isdir(dos_save):
    os.mkdir(dos_save)

MAXTHROTTLE = 0.5
th_direction = 0.05
th_throttle = 0.06

comPort = "/dev/ttyUSB0"
ser = serial_command2.start_serial(comPort)
joy = xbox.Joystick()
cap = cv2.VideoCapture(0)

basedir = os.path.dirname(os.path.abspath(__file__))
model = architectures.load_model(
    os.path.normpath(f'{basedir}/../test_model/models/test_home.h5'))
architectures.apply_predict_decorator(model)

prev_throttle = 0
while not joy.Back():
    joy_steering = joy.leftX()
    joy_throttle = joy.rightTrigger()
    joy_brake = joy.leftTrigger()
    joy_button_a = joy.A()
    joy_button_x = joy.X()

    _, cam = cap.read()
    img = cv2.resize(cam, (160, 120))

    # annotation template with just what is needed for the prediction
    annotation = {
        'direction': 0,
        'speed': prev_throttle,
        'throttle': 0,
        'time': time.time()
    }

    if joy_button_a or joy_button_x:

        steering = joy_steering if abs(joy_steering) > abs(th_direction) else 0
        throttle = joy_throttle - joy_brake if abs(joy_throttle - joy_brake) > abs(th_throttle) else 0

        if not joy_button_x:
            Dataset.save_img_and_annotation(
                img,
                {
                    'direction': float(steering),
                    'speed': float(prev_throttle),
                    'throttle': float(throttle),
                    'time': time.time()
                },
                './recorded/')

        ser.ChangeAll(steering, MAXTHROTTLE * throttle)

    else:
        annotation_list = drive_utils.dict2list(annotation)
        to_pred = Dataset.make_to_pred_annotations(
            [img], [annotation_list], input_components)

        predicted, dt = model.predict(to_pred)
        predicted = predicted[0]

        steering = predicted['direction'][0]
        throttle = predicted['throttle'][0]

        ser.ChangeAll(steering, MAXTHROTTLE * throttle)

    prev_throttle = throttle


joy.close()
print('terminated')
