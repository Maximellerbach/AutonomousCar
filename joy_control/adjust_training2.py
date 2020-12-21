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

comPort = drive_utils.get_port_name()
ser = serial_command2.start_serial(comPort)
joy = xbox.Joystick()
cap = cv2.VideoCapture(0)

basedir = os.path.dirname(os.path.abspath(__file__))
model = architectures.load_model(
    os.path.normpath(f'{basedir}/../test_model/models/rbrl_sim5_working.h5'))
architectures.apply_predict_decorator(model)

prev_throttle = 0
while not joy.Back():
    joy_steering = joy.leftX()
    joy_throttle = joy.rightTrigger()
    joy_brake = joy.leftTrigger()
    joy_button_a = joy.A()
    joy_button_x = joy.X()

    _, img = cap.read()

    # annotation template with just what is needed for the prediction
    annotation = {
        'direction': 0,
        'speed': prev_throttle,
        'throttle': 0,
        'time': time.time()
    }

    if joy_button_a or joy_button_x:

        steering = joy_steering if abs(joy_steering) > abs(
            th_direction) and not joy_button_x else 0
        throttle = joy_throttle - \
            joy_brake if abs(joy_throttle) > abs(th_throttle) else 0

        annotation['direction'] = steering
        annotation['throttle'] = throttle

        if not joy_button_x:
            Dataset.save_img_and_annotation(img, annotation, './recorded/')

    else:
        annotation_list = drive_utils.dict2list(annotation)
        to_pred = Dataset.make_to_pred_annotations(
            [img], [annotation_list], input_components)

        output_dict, elapsed_time = model.predict(to_pred)
        output_dict = output_dict[0]
        for key in output_dict:
            annotation[key] = output_dict[key]

    ser.ChangeAll(annotation['direction'], MAXTHROTTLE * annotation['throttle'], min=[-1, -1], max=[1, 1])


joy.close()
print('terminated')
