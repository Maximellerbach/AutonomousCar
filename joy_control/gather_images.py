import getopt
import os
import sys
import time

import cv2

import xbox
from custom_modules import serial_command
from custom_modules.datasets import dataset_json


def printusage():
    print("python3 " + os.path.basename(__file__) + " -c <COM_Port>")
    print("<COM_Port>")
    print("  Windows: COMx where x is a number")
    print("  Linux: /dev/ttyXYZ where XYZ can be S2 or USB0 or any valid tty type")
    print("         keep in mind you need root priviledges to acces serial ports in Linux")
    print("         so execute the command in priviledge mode like: sudo python3 " +
          os.path.basename(__file__) + " -c /dev/ttyS2")
    print("  You can use as well environement variable COM_PORT. Arguments will prevent on environment variable")
    print("To install Xbox drivers on RPI: sudo apt-get install xboxdrv")


def get_port_name():
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
        else:
            printusage()
            sys.exit(2)

    if (comPort == ""):
        try:
            comPort = os.environ["COM_PORT"]
        except KeyError:
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


Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
dos_save = os.getcwd()+os.path.normpath("/recorded/")
if not os.path.isdir(dos_save):
    os.mkdir(dos_save)

MAXSPEED = 120
th_direction = 0.05
th_throttle = 0.1

comPort = get_port_name()
ser = serial_command.start_serial(comPort)
joy = xbox.Joystick()

cap = cv2.VideoCapture(0)

while not joy.Back():
    joy_steering, joy_throttle, joy_brake, joy_button_a, joy_button_x = get_controller_buttons(
        joy)

    steering = joy_steering if abs(joy_steering) > abs(
        th_direction) and not joy_button_x else 0
    throttle = joy_throttle - \
        joy_brake if abs(joy_throttle) > abs(th_throttle) else 0
    current_speed = ser.GetCurrentSpeed()

    pwm = int(MAXSPEED * throttle)
    ser.ChangeMotorA(serial_command.motor.MOTOR_BACKWARD)
    ser.ChangePWM(pwm)

    if joy_button_a:
        _, img = cap.read()
        Dataset.save_img_and_annotation(
            img,
            [steering, current_speed, throttle, time.time()],
            dos=dos_save)


joy.close()
print('terminated')
