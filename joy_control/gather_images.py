import cv2
import os
import time

import xbox
from custom_modules import serial_command, drive_utils
from custom_modules.datasets import dataset_json

Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
dos_save = os.getcwd()+os.path.normpath("/recorded/")
if not os.path.isdir(dos_save):
    os.mkdir(dos_save)

MAXTHROTTLE = 120
th_direction = 0.05
th_throttle = 0.1

comPort = drive_utils.get_port_name()
ser = serial_command.start_serial(comPort)
joy = xbox.Joystick()
cap = cv2.VideoCapture(0)

ser.ChangeMotorA(serial_command.Motor.MOTOR_FORWARD)
while not joy.Back():
    joy_steering, joy_throttle, joy_brake, joy_button_a, joy_button_x = drive_utils.get_controller_buttons(
        joy)

    steering = joy_steering if abs(joy_steering) > abs(
        th_direction) and not joy_button_x else 0
    throttle = joy_throttle - \
        joy_brake if abs(joy_throttle) > abs(th_throttle) else 0
    current_speed = ser.GetCurrentSpeed()

    pwm = int(MAXTHROTTLE * throttle)
    ser.ChangeDirection(drive_utils.direction2categorical(steering))
    ser.ChangePWM(pwm)

    if joy_button_a:
        _, img = cap.read()

        Dataset.save_img_and_annotation(
            img,
            {
                'direction': steering,
                'speed': current_speed,
                'throttle': throttle,
                'time': time.time()
            }
        )


joy.close()
print('terminated')
