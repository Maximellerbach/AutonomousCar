import os
import time

import cv2
from custom_modules import serial_command2, camera, memory, pid_controller
from custom_modules.datasets import dataset_json

import controller


def crop_image(img):
    return img[20: 120].copy()


def deadzone(value, th, default=0):
    return value if abs(value) > th else default


dos_save = os.path.expanduser("~") + "/recorded/"
if not os.path.isdir(dos_save):
    os.mkdir(dos_save)

Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])

Memory = memory.Memory(Dataset, dos_save, queue_size=10)
Memory.run()

MAXTHROTTLE = 1
th_steering = 0.05  # 5% threshold
th_throttle = 0.06  # 6% threshold

comPort = "/dev/ttyUSB0"
ser = serial_command2.start_serial(comPort)

joy = controller.XboxOneJoystick()
joy.init()
assert joy.connected is True

# cap = camera.usbWebcam(topcrop=0.2, botcrop=0)
cap = cv2.VideoCapture(0)
# cap.start()

# checking if the controller is working properly
joy_leftX = 0
while joy_leftX <= 0.9:
    joy_leftX = joy.axis_states["x"]
    print(joy_leftX, end="\r")
    time.sleep(0.01)

while joy_leftX >= -0.9:
    joy_leftX = joy.axis_states["x"]
    print(joy_leftX, end="\r")
    time.sleep(0.01)

print("Starting mainloop")

while not joy.button_states["back"] and joy.connected:
    joy_steering = joy.axis_states["x"]
    joy_throttle = joy.axis_states["rz"]
    joy_brake = joy.axis_states["z"]
    joy_button_a = joy.button_states["a"]

    annotation = {}

    annotation["direction"] = deadzone(joy_steering, th_steering)
    annotation["throttle"] = deadzone(joy_throttle - joy_brake, th_throttle)
    annotation["speed"] = ser.GetSpeed()
    annotation["time"] = time.time()

    ser.ChangeAll(annotation["direction"], MAXTHROTTLE *
                  annotation["throttle"], min=[-1, -1], max=[1, 1])

    # print("speed", annotation["speed"])
    # print("throttle", annotation["throttle"])

    if joy_button_a:  # save the image
        _, img = cap.read()
        img = cv2.resize(img, (160, 120))

        Memory.add(img, annotation)

    time.sleep(0.1)

Memory.stop()
ser.ChangeAll(0, 0)  # stop steering and throttle
cap.release()

if not joy.connected:
    print("Lost connection with joystick")
else:
    print("Terminated")
