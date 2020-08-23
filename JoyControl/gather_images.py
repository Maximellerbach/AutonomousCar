import getopt
import os
import sys
import time

import cv2

import xbox
from custom_modules import serialCommand, DatasetJson


def get_args():
    def printusage():
        print(__file__ + " -c <COM_Port>")
        print("  Windows: COMx where x is a number")
        print("  Linux: /dev/ttyXYZ where XYZ can be S2 or USB0")
        print("To install Xbox drivers on RPI: sudo apt-get install xboxdrv")

    comPort = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:h", ["com", "help"])
    except getopt.GetoptError:
        printusage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            printusage()
            sys.exit()
        elif opt in ("-c", "--com"):
            comPort = arg
            # continue the program
        else:
            printusage()
            sys.exit(2)

    return comPort


def get_controller_buttons(joy):
    direction = joy.leftX()
    throttle = joy.rightTrigger()
    brake = joy.leftTrigger()
    button_a = joy.A()
    button_x = joy.X()
    return (direction, throttle, brake, button_a, button_x)


Dataset = DatasetJson(["direction", "speed", "throttle", "time"])
dos_save = os.getcwd()+os.path.normpath("/recorded/")
if not os.path.isdir(dos_save):
    os.mkdir(dos_save)

MAXSPEED = 120
th_direction = 0.05
th_throttle = 0.1

comPort = get_args()
ser = serialCommand.start_serial(comPort)
joy = xbox.Joystick()

cap = cv2.VideoCapture(0)

while not joy.Back():
    joy_steering, joy_throttle, joy_brake, joy_button_a, joy_button_x = get_controller_buttons(
        joy)

    steering = joy_steering if abs(joy_steering) > abs(th_direction) and not joy_button_x else 0
    throttle = joy_throttle - joy_brake if abs(joy_throttle) > abs(th_throttle) else 0
    current_speed = ser.GetCurrentSpeed()

    pwm = int(MAXSPEED * throttle)
    ser.ChangeMotorA(serialCommand.motor.MOTOR_BACKWARD)
    ser.ChangePWM(pwm)

    if joy_button_a:
        _, img = cap.read()
        Dataset.save_img_and_annotations(
            img,
            [steering, current_speed, throttle, time.time()],
            dos=dos_save)


joy.close()
print('terminated')
