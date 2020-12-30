import os
import time

from custom_modules import serial_command2, pid_steering
from custom_modules.datasets import dataset_json

import cv2
import xbox


def deadzone(value, th, default=0):
    return value if abs(value) > th else default


Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
dos_save = os.getcwd()+os.path.normpath("/recorded/")
if not os.path.isdir(dos_save):
    os.mkdir(dos_save)

pidSteering = pid_steering.SimpleSteering()

MAXTHROTTLE = 0.5
th_position = 0.05  # 5% threshold
th_throttle = 0.06  # 6% threshold

comPort = "/dev/ttyUSB0"
ser = serial_command2.start_serial(comPort)
joy = xbox.Joystick()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

print(joy.connected())

prev_throttle = 0
while not joy.Back():
    joy_position = joy.leftX()
    joy_throttle = joy.rightTrigger()
    joy_brake = joy.leftTrigger()
    joy_button_a = joy.A()

    position = deadzone(joy_position, th_position)
    throttle = deadzone(joy_throttle - joy_brake, th_throttle)
    last_received = time.time()

    steering = pidSteering.update_steering(position, last_received=last_received)
    ser.ChangeAll(steering, MAXTHROTTLE * throttle, min=[-1, -1], max=[1, 1])

    if joy_button_a:
        _, img = cap.read()

        Dataset.save_img_and_annotation(
            img,
            annotation={
                'direction': steering,
                'speed': prev_throttle,  # save previous throttle
                'throttle': throttle,  # save raw throttle value
                'time': last_received
            },
            dos=dos_save
        )
    prev_throttle = throttle

print('terminated')
