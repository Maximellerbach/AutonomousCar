import os
import time

from custom_modules import serial_command2
from custom_modules.datasets import dataset_json


import xbox

Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
dos_save = os.getcwd()+os.path.normpath("/recorded/")
if not os.path.isdir(dos_save):
    os.mkdir(dos_save)

MAXTHROTTLE = 0.5
th_direction = 0.05  # 5% threshold
th_throttle = 0.06  # 6% threshold

comPort = "/dev/ttyUSB0"
ser = serial_command2.start_serial(comPort)
joy = xbox.Joystick()
# cap = cv2.VideoCapture(0)

print("I'm HERE !")
print(joy.connected())

prev_throttle = 0
while not joy.Back():
    joy_steering = joy.leftX()
    joy_throttle = joy.rightTrigger()
    joy_brake = joy.leftTrigger()
    joy_button_a = joy.A()
    joy_button_x = joy.X()

    steering = joy_steering if abs(joy_steering) > abs(th_direction) and not joy_button_x else 0
    throttle = joy_throttle - joy_brake if abs(joy_throttle - joy_brake) > abs(th_throttle) else 0

    pwm = MAXTHROTTLE * throttle
    ser.ChangeAll(steering, pwm, min=[-1, -1], max=[1, 1])

    # if joy_button_a:
    #     _, img = cap.read()

    #     Dataset.save_img_and_annotation(
    #         img,
    #         {
    #             'direction': steering,
    #             'speed': prev_throttle,  # save previous throttle
    #             'throttle': throttle,  # save raw throttle value
    #             'time': time.time()
    #         }
    #     )
    prev_throttle = throttle

print('terminated')
